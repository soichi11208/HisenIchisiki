#include "search.h"
#include "../engine/kiki.h"
#include "../engine/mate.h"
#include <ATen/autocast_mode.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

namespace hisen {

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

Evaluator::Result Evaluator::evaluate(const std::vector<const Graph*>& graphs) {
  torch::NoGradGuard ng;
  auto batch = GraphBatch::collate(graphs).to(device);
  bool useAmp = amp && device.is_cuda();
  if (useAmp) {
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
  }
  auto [logits, values] = net->forward(batch);
  if (useAmp) {
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    at::autocast::clear_cache();
  }
  auto probsGpu = torch::cat(splitPolicyLogSoftmax(logits, batch.moveCounts)).exp();
  auto probsCpu = probsGpu.to(torch::kCPU).to(torch::kFloat).contiguous();
  auto vCpu = values.to(torch::kCPU).to(torch::kFloat).contiguous();
  Result res;
  res.policies.reserve(graphs.size());
  res.values.reserve(graphs.size());
  const float* pp = probsCpu.data_ptr<float>();
  const float* vp = vCpu.data_ptr<float>();
  int64_t off = 0;
  for (size_t i = 0; i < graphs.size(); i++) {
    int64_t m = batch.moveCounts[i];
    res.policies.emplace_back(pp + off, pp + off + m);
    res.values.push_back(vp[i]);
    off += m;
  }
  return res;
}

BatchedEvaluator::BatchedEvaluator(HisenNet net, torch::Device dev, int maxBatch,
                                   bool useAmp)
    : Evaluator(std::move(net), dev, useAmp), maxBatch_(maxBatch) {
  worker_ = std::thread([this] { workerLoop(); });
}

BatchedEvaluator::~BatchedEvaluator() {
  { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

Evaluator::Result BatchedEvaluator::evaluate(const std::vector<const Graph*>& graphs) {
  Req req;
  req.graphs = graphs;
  auto fut = req.prom.get_future();
  {
    std::lock_guard<std::mutex> lk(mtx_);
    queue_.push(&req);
  }
  cv_.notify_one();
  return fut.get();
}

void BatchedEvaluator::workerLoop() {
  while (true) {
    std::vector<Req*> batch;
    int total = 0;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_ && queue_.empty()) return;
      while (!queue_.empty()) {
        int n = int(queue_.front()->graphs.size());
        if (!batch.empty() && total + n > maxBatch_) break;
        batch.push_back(queue_.front());
        queue_.pop();
        total += n;
        if (total >= maxBatch_) break;
      }
    }
    std::vector<const Graph*> all;
    all.reserve(total);
    for (Req* r : batch)
      for (const Graph* g : r->graphs) all.push_back(g);
    Result res = Evaluator::evaluate(all);
    size_t off = 0;
    for (Req* r : batch) {
      Result sub;
      size_t n = r->graphs.size();
      sub.policies.assign(std::make_move_iterator(res.policies.begin() + off),
                          std::make_move_iterator(res.policies.begin() + off + n));
      sub.values.assign(res.values.begin() + off, res.values.begin() + off + n);
      off += n;
      r->prom.set_value(std::move(sub));
    }
  }
}

// ---------------------------------------------------------------------------
// MctsNode
// ---------------------------------------------------------------------------

int MctsNode::totalN() const {
  int n = 0;
  for (size_t i = 0; i < N.size(); i++) n += N[i] + VL[i];
  return n;
}

int MctsNode::selectChild(float cPuct, float fpuReduction) const {
  float sqrtTotal = std::sqrt(float(totalN()) + 1e-8f);
  int totalN_ = 0;
  float totalW_ = 0.f;
  for (size_t i = 0; i < N.size(); i++) {
    totalN_ += N[i];
    totalW_ += W[i];
  }
  float parentQ = totalN_ > 0 ? totalW_ / totalN_ : 0.f;
  float fpuQ = parentQ - fpuReduction;
  int best = 0;
  float bestScore = -1e30f;
  for (size_t i = 0; i < moves.size(); i++) {
    int n = N[i] + VL[i];
    float q = n > 0 ? (W[i] - VL[i]) / n : fpuQ;
    const MctsNode* c = children[i].get();
    // 証明済みの勝ち手は最優先、負け手は避ける
    if (c && c->terminal) {
      if (c->terminalValue <= -0.5f) return int(i); // 相手負け = 自分勝ち
      q = -c->terminalValue;
    }
    float u = cPuct * P[i] * sqrtTotal / (1 + n);
    if (q + u > bestScore) {
      bestScore = q + u;
      best = int(i);
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Search helpers
// ---------------------------------------------------------------------------

namespace {

// 詰み探索の暴走防止: 1 回のルート/リーフ詰み探索で使う最大ノード数。
// 時間制御がある場合は締切 (MateAbort.deadlineTicks) でも打ち切る。
constexpr int64_t kMateRootNodeBudget = 400000;
constexpr int64_t kMateLeafNodeBudget = 20000;

void expand(MctsNode& node, const Graph& g, const std::vector<float>& policy) {
  node.moves = g.moves;
  node.P = policy;
  size_t n = node.moves.size();
  node.N.assign(n, 0);
  node.W.assign(n, 0.f);
  node.VL.assign(n, 0);
  node.children.resize(n);
  node.expanded = true;
}

float backupStep(MctsNode* parent, int childIdx, float v) {
  parent->VL[childIdx]--;
  parent->N[childIdx]++;
  parent->W[childIdx] += v;
  const MctsNode* child = parent->children[childIdx].get();
  if (!child || !child->terminal || parent->terminal) return v;
  if (child->terminalValue <= -0.5f) {
    parent->terminal = true;
    parent->terminalValue = +1.f;
    return +1.f;
  }
  if (child->terminalValue >= +0.5f) {
    for (auto& c : parent->children) {
      if (!c || !c->terminal || c->terminalValue < +0.5f) return v;
    }
    parent->terminal = true;
    parent->terminalValue = -1.f;
    return -1.f;
  }
  return v;
}

int matPieceValue(PieceType pt) {
  switch (rawType(pt)) {
    case PAWN: return 1;
    case LANCE:
    case KNIGHT: return 3;
    case SILVER: return 5;
    case GOLD: return 6;
    case BISHOP: return 8;
    case ROOK: return 10;
    case KING: return 0;
    default: return isPromoted(pt) ? 6 : 0;
  }
}

// 手番側視点の簡易駒得評価を [-1,1] に圧縮 (戦術 qsearch の stand-pat 補助)
float materialEval(const Position& pos) {
  int s = 0;
  for (Square sq = 0; sq < SQ_NB; sq++) {
    Piece p = pos.board[sq];
    if (!p) continue;
    int v = matPieceValue(typeOf(p));
    s += (colorOf(p) == pos.side) ? v : -v;
  }
  for (int c = 0; c < 2; c++) {
    for (int t = PAWN; t <= ROOK; t++) {
      int v = matPieceValue(PieceType(t)) * pos.hand[c][t];
      s += (Color(c) == pos.side) ? v : -v;
    }
  }
  // 約 1 枚の飛車差で |eval|≈0.5
  return std::tanh(float(s) / 12.f);
}

bool isCaptureMove(const Position& pos, Move m) {
  return !isDrop(m) && pos.board[moveTo(m)] != 0;
}

bool givesCheck(const Position& pos, Move m) {
  Position next = pos;
  next.doMove(m);
  return inCheck(next, next.side);
}

} // namespace

Search::Search(Evaluator& eval, SearchParams params) : eval_(eval), params_(params) {
  rng_.seed(params.seed ? params.seed : std::random_device{}());
}

void Search::clearTree() {
  root_.reset();
  haveTree_ = false;
  provenMateMove_ = MOVE_NONE;
  lastReuseHit_ = false;
  {
    std::lock_guard<std::mutex> lk(cacheMtx_);
    evalCache_.clear();
  }
  clearMateTable();
}

void Search::clearVirtualLoss(MctsNode* node) {
  if (!node || !node->expanded) return;
  std::fill(node->VL.begin(), node->VL.end(), 0);
  for (auto& c : node->children)
    if (c) clearVirtualLoss(c.get());
}

bool Search::findPathTo(MctsNode* node, Position pos, const Position& target,
                        std::vector<int>& path, int maxDepth) const {
  if (pos == target) return true;
  if (maxDepth <= 0 || !node || !node->expanded) return false;
  for (size_t i = 0; i < node->moves.size(); i++) {
    Position next = pos;
    next.doMove(node->moves[i]);
    path.push_back(int(i));
    // 子が未展開でも next==target ならここに降りられる
    if (next == target) return true;
    MctsNode* child = node->children[i].get();
    if (findPathTo(child, next, target, path, maxDepth - 1)) return true;
    path.pop_back();
  }
  return false;
}

bool Search::tryReuseRoot(const Position& target) {
  if (!haveTree_ || !root_ || !root_->expanded) return false;
  if (rootPos_ == target) {
    clearVirtualLoss(root_.get());
    return true;
  }
  std::vector<int> path;
  if (!findPathTo(root_.get(), rootPos_, target, path, /*maxDepth=*/8)) return false;
  for (int idx : path) {
    if (!root_->expanded || idx < 0 || idx >= int(root_->moves.size())) return false;
    Move m = root_->moves[idx];
    std::unique_ptr<MctsNode> child = std::move(root_->children[idx]);
    if (!child) child = std::make_unique<MctsNode>();
    root_ = std::move(child);
    rootPos_.doMove(m);
  }
  // 目標局面まで降りられた (未展開でもルート差し替え成功 = 再利用ヒット)
  clearVirtualLoss(root_.get());
  return true;
}

void Search::addDirichletNoise(MctsNode& node) {
  if (params_.dirichletEps <= 0 || node.P.empty()) return;
  std::gamma_distribution<float> gamma(params_.dirichletAlpha, 1.f);
  std::vector<float> noise(node.P.size());
  float sum = 0;
  for (auto& x : noise) {
    x = gamma(rng_);
    sum += x;
  }
  if (sum <= 0) return;
  for (size_t i = 0; i < node.P.size(); i++)
    node.P[i] = (1 - params_.dirichletEps) * node.P[i] +
                params_.dirichletEps * noise[i] / sum;
}

bool Search::cacheLookup(uint64_t key, const std::vector<Move>& moves,
                         std::vector<float>& policy, float& value) const {
  if (params_.evalCacheSize == 0) return false;
  std::lock_guard<std::mutex> lk(cacheMtx_);
  auto it = evalCache_.find(key);
  if (it == evalCache_.end()) return false;
  if (it->second.moves != moves) return false; // 合法手集合/順序が違う
  policy = it->second.policy;
  value = it->second.value;
  return true;
}

void Search::cacheStore(uint64_t key, const std::vector<Move>& moves,
                        const std::vector<float>& policy, float value) {
  if (params_.evalCacheSize == 0) return;
  std::lock_guard<std::mutex> lk(cacheMtx_);
  if (evalCache_.size() >= params_.evalCacheSize) {
    // 簡易: 半分捨てる (完全 LRU は重い)
    auto it = evalCache_.begin();
    size_t n = evalCache_.size() / 2;
    for (size_t i = 0; i < n && it != evalCache_.end();) {
      it = evalCache_.erase(it);
      ++i;
    }
  }
  evalCache_[key] = CacheEntry{moves, policy, value};
}

float Search::leafTactical(const Position& pos, float nnValue) const {
  const int depth = params_.tacticalDepth;
  if (depth <= 0) return nnValue;

  // ネスト qsearch: 捕獲・王手のみ。stand-pat は nn または material。
  std::function<float(const Position&, int, float, float, float)> qs =
      [&](const Position& p, int d, float alpha, float beta, float stand) -> float {
        if (stand >= beta) return stand;
        if (stand > alpha) alpha = stand;
        if (d <= 0) return alpha;

        auto moves = legalMoves(p);
        if (moves.empty()) return inCheck(p, p.side) ? -1.f : 0.f;

        struct Scored {
          int score;
          Move m;
        };
        std::vector<Scored> tacs;
        tacs.reserve(moves.size());
        for (Move m : moves) {
          bool cap = isCaptureMove(p, m);
          bool chk = givesCheck(p, m);
          if (!cap && !chk) continue;
          int sc = 0;
          if (cap) sc += 10 * matPieceValue(typeOf(p.board[moveTo(m)]));
          if (chk) sc += 50;
          if (isPromo(m)) sc += 8;
          tacs.push_back({sc, m});
        }
        if (tacs.empty()) return alpha;
        std::sort(tacs.begin(), tacs.end(),
                  [](const Scored& a, const Scored& b) { return a.score > b.score; });

        for (auto& t : tacs) {
          Position next = p;
          next.doMove(t.m);
          // 即座に詰み
          if (legalMoves(next).empty() && inCheck(next, next.side)) {
            if (1.f >= beta) return 1.f;
            if (1.f > alpha) alpha = 1.f;
            continue;
          }
          float childStand = materialEval(next);
          float score = -qs(next, d - 1, -beta, -alpha, childStand);
          if (score >= beta) return score;
          if (score > alpha) alpha = score;
        }
        return alpha;
      };

  float v = qs(pos, depth, -1.f, 1.f, nnValue);
  // nn と戦術結果をブレンド (極端な戦術結果は寄せる)
  if (v > 0.95f || v < -0.95f) return v;
  return 0.65f * nnValue + 0.35f * v;
}

void Search::evaluateLeaves(const std::vector<const Graph*>& graphs,
                            const std::vector<uint64_t>& keys,
                            const std::vector<Position>& positions,
                            std::vector<std::vector<float>>& policies,
                            std::vector<float>& values) {
  const size_t n = graphs.size();
  policies.assign(n, {});
  values.assign(n, 0.f);
  if (n == 0) return;

  // キャッシュヒット / 同一局面合流
  std::vector<int> needIdx;
  needIdx.reserve(n);
  std::unordered_map<uint64_t, int> firstOfKey; // key → 代表インデックス
  std::vector<int> repOf(n, -1);               // 同一 key は代表へ合流

  for (size_t i = 0; i < n; i++) {
    std::vector<float> pol;
    float val = 0.f;
    if (cacheLookup(keys[i], graphs[i]->moves, pol, val)) {
      policies[i] = std::move(pol);
      values[i] = leafTactical(positions[i], val);
      continue;
    }
    auto it = firstOfKey.find(keys[i]);
    if (it != firstOfKey.end()) {
      repOf[i] = it->second;
      continue;
    }
    firstOfKey[keys[i]] = int(i);
    repOf[i] = int(i);
    needIdx.push_back(int(i));
  }

  if (!needIdx.empty()) {
    std::vector<const Graph*> uniq;
    uniq.reserve(needIdx.size());
    for (int i : needIdx) uniq.push_back(graphs[i]);
    auto res = eval_.evaluate(uniq);
    for (size_t u = 0; u < needIdx.size(); u++) {
      int i = needIdx[u];
      policies[i] = res.policies[u];
      float nnV = res.values[u];
      cacheStore(keys[i], graphs[i]->moves, policies[i], nnV);
      values[i] = leafTactical(positions[i], nnV);
    }
  }

  // 合流コピー
  for (size_t i = 0; i < n; i++) {
    if (repOf[i] >= 0 && repOf[i] != int(i) && policies[i].empty()) {
      int r = repOf[i];
      policies[i] = policies[r];
      values[i] = values[r]; // 同一局面なので tactical も同じ
    }
  }
}

void Search::topTwoVisits(const MctsNode& root, int& best, int& second) const {
  best = 0;
  second = 0;
  for (int n : root.N) {
    if (n > best) {
      second = best;
      best = n;
    } else if (n > second) {
      second = n;
    }
  }
}

bool Search::isVisitStable(const MctsNode& root) const {
  if (!root.expanded || root.N.empty()) return false;
  int best = 0, second = 0;
  topTwoVisits(root, best, second);
  if (best < params_.minVisitsForStop) return false;
  if (second <= 0) return best >= params_.minVisitsForStop;
  return float(best) >= float(second) * params_.stableVisitRatio;
}

// ---------------------------------------------------------------------------
// Main search
// ---------------------------------------------------------------------------

MctsNode* Search::run(const Position& rootPos, const std::vector<uint64_t>& history,
                      const std::atomic<bool>* stopFlag, Clock::time_point hardDeadline,
                      Clock::time_point softDeadline) {
  bool hasHard = hardDeadline != Clock::time_point{};
  bool hasSoft = softDeadline != Clock::time_point{};
  if (hasHard && !hasSoft) softDeadline = hardDeadline; // soft 未指定なら hard と同一

  clearMateTable();
  provenMateMove_ = MOVE_NONE;
  lastReuseHit_ = false;

  // --- ツリー再利用 ---
  bool reused = tryReuseRoot(rootPos);
  lastReuseHit_ = reused;
  if (!reused) {
    root_ = std::make_unique<MctsNode>();
    rootPos_ = rootPos;
    Graph g = buildGraph(rootPos);
    if (g.moves.empty()) {
      root_->terminal = true;
      root_->terminalValue = -1.f;
      root_->expanded = true;
      haveTree_ = true;
      return root_.get();
    }
    uint64_t key = rootPos.hashKey();
    std::vector<std::vector<float>> pols;
    std::vector<float> vals;
    evaluateLeaves({&g}, {key}, {rootPos}, pols, vals);
    expand(*root_, g, pols[0]);
    addDirichletNoise(*root_);
    haveTree_ = true;
  } else {
    rootPos_ = rootPos;
    if (!root_->expanded) {
      Graph g = buildGraph(rootPos);
      if (g.moves.empty()) {
        root_->terminal = true;
        root_->terminalValue = -1.f;
        root_->expanded = true;
        return root_.get();
      }
      std::vector<std::vector<float>> pols;
      std::vector<float> vals;
      evaluateLeaves({&g}, {rootPos.hashKey()}, {rootPos}, pols, vals);
      expand(*root_, g, pols[0]);
    }
    // 再利用時は Dirichlet を掛けない (方策を壊さない)
  }

  // ルート詰み (攻め)
  if (params_.mateDepth > 0) {
    Move mm;
    MateAbort ab;
    ab.stopFlag = stopFlag;
    ab.deadlineTicks = hasHard ? hardDeadline.time_since_epoch().count() : 0;
    ab.nodeBudget = kMateRootNodeBudget;
    if (canForceMate(rootPos, params_.mateDepth, &mm, &ab)) provenMateMove_ = mm;
  }

  // 既に証明済み or 詰み手確定なら即返す
  if (provenMateMove_ != MOVE_NONE || root_->terminal) return root_.get();

  if (params_.threads > 1) {
    runParallel(rootPos, history, stopFlag, hardDeadline, softDeadline, hasHard, hasSoft);
    return root_.get();
  }

  struct PathStep {
    MctsNode* node;
    int child;
  };
  int done = 0;
  auto shouldStop = [&] {
    if (stopFlag && stopFlag->load()) return true;
    if (root_->terminal) return true;
    auto now = Clock::now();
    if (hasHard && now >= hardDeadline) return true;
    if (hasSoft && now >= softDeadline && isVisitStable(*root_)) return true;
    return false;
  };

  // リーフ詰み探索の打ち切り条件 (シリアル探索経路)
  MateAbort leafAbort;
  leafAbort.stopFlag = stopFlag;
  leafAbort.deadlineTicks = hasHard ? hardDeadline.time_since_epoch().count() : 0;
  leafAbort.nodeBudget = kMateLeafNodeBudget;

  while (done < params_.simulations) {
    if (shouldStop()) break;

    struct Pending {
      std::vector<PathStep> path;
      MctsNode* leaf;
      Graph graph;
      Position pos;
      uint64_t key;
    };
    std::vector<Pending> pendings;
    int want = std::min(params_.batchSize, params_.simulations - done);

    for (int b = 0; b < want; b++) {
      if (shouldStop()) break;
      MctsNode* node = root_.get();
      Position pos = rootPos;
      std::vector<PathStep> path;
      std::vector<uint64_t> seen = history;
      seen.push_back(pos.hashKey());
      bool repetition = false;

      while (node->expanded && !node->terminal) {
        int c = node->selectChild(params_.cPuct, params_.fpuReduction);
        node->VL[c]++;
        path.push_back({node, c});
        pos.doMove(node->moves[c]);
        uint64_t key = pos.hashKey();
        int cnt = int(std::count(seen.begin(), seen.end(), key));
        if (cnt + 1 >= 4) {
          repetition = true;
          break;
        }
        seen.push_back(key);
        if (!node->children[c]) node->children[c] = std::make_unique<MctsNode>();
        node = node->children[c].get();
      }

      auto backup = [&](float leafValue) {
        float v = leafValue;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
          v = -v;
          v = backupStep(it->node, it->child, v);
        }
        done++;
      };

      if (repetition) {
        backup(0.f);
        continue;
      }
      if (node->terminal) {
        backup(node->terminalValue);
        continue;
      }

      Graph g = buildGraph(pos);
      if (g.moves.empty()) {
        node->terminal = true;
        node->terminalValue = -1.f;
        node->expanded = true;
        backup(-1.f);
        continue;
      }
      if (params_.mateDepth > 0 &&
          canForceMate(pos, params_.mateDepth, nullptr, &leafAbort)) {
        node->terminal = true;
        node->terminalValue = +1.f;
        node->expanded = true;
        backup(+1.f);
        continue;
      }
      pendings.push_back({std::move(path), node, std::move(g), pos, pos.hashKey()});
    }

    if (!pendings.empty()) {
      std::vector<const Graph*> graphs;
      std::vector<uint64_t> keys;
      std::vector<Position> positions;
      graphs.reserve(pendings.size());
      for (auto& p : pendings) {
        graphs.push_back(&p.graph);
        keys.push_back(p.key);
        positions.push_back(p.pos);
      }
      std::vector<std::vector<float>> pols;
      std::vector<float> vals;
      evaluateLeaves(graphs, keys, positions, pols, vals);
      for (size_t i = 0; i < pendings.size(); i++) {
        auto& p = pendings[i];
        if (!p.leaf->expanded) expand(*p.leaf, p.graph, pols[i]);
        float v = vals[i];
        for (auto it = p.path.rbegin(); it != p.path.rend(); ++it) {
          v = -v;
          v = backupStep(it->node, it->child, v);
        }
        done++;
      }
    }

    // ルートで勝ち証明されたら打ち切り
    if (root_->terminal) break;
  }
  return root_.get();
}

void Search::runParallel(const Position& rootPos, const std::vector<uint64_t>& history,
                         const std::atomic<bool>* stopFlag, Clock::time_point hardDeadline,
                         Clock::time_point softDeadline, bool hasHard, bool hasSoft) {
  struct PathStep {
    MctsNode* node;
    int child;
  };
  std::mutex treeMtx;

  struct EvalReq {
    const Graph* g;
    uint64_t key;
    Position pos;
    std::promise<std::pair<std::vector<float>, float>> prom;
  };
  std::mutex qMtx;
  std::condition_variable qCv;
  std::queue<EvalReq*> queue;
  bool evalStop = false;
  // 例外はスレッド内で捕捉して停止する (std::terminate = プロセス即死を防止)。
  std::atomic<bool> evalDead{false};
  std::atomic<bool> errReported{false};
  auto reportErr = [&](const std::string& what) {
    if (errReported.exchange(true)) return;
    std::string line = "info string ERROR search worker/eval exception: " + what +
                       " -- stopping early";
    std::cout << line << std::endl;
  };
  // 評価器が死んだ後のフォールバック: 一律 policy, value 0
  auto fulfillFallback = [](std::vector<EvalReq*>& reqs) {
    for (auto* r : reqs) {
      size_t m = r->g ? r->g->moves.size() : 0;
      if (m == 0) m = 1;
      r->prom.set_value({std::vector<float>(m, 1.f / float(m)), 0.f});
    }
  };
  // 呼び出し側 (main) は非 const のフラグを所有している
  auto requestStop = [&] {
    if (stopFlag) const_cast<std::atomic<bool>*>(stopFlag)->store(true);
  };

  std::thread evalThread([&] {
    while (true) {
      std::vector<EvalReq*> batch;
      {
        std::unique_lock<std::mutex> lk(qMtx);
        qCv.wait(lk, [&] { return evalStop || !queue.empty(); });
        if (evalStop && queue.empty()) return;
        while (!queue.empty() && int(batch.size()) < params_.batchSize) {
          batch.push_back(queue.front());
          queue.pop();
        }
      }
      try {
        std::vector<const Graph*> gs;
        std::vector<uint64_t> keys;
        std::vector<Position> positions;
        gs.reserve(batch.size());
        for (auto* r : batch) {
          gs.push_back(r->g);
          keys.push_back(r->key);
          positions.push_back(r->pos);
        }
        std::vector<std::vector<float>> pols;
        std::vector<float> vals;
        evaluateLeaves(gs, keys, positions, pols, vals);
        for (size_t i = 0; i < batch.size(); i++)
          batch[i]->prom.set_value({std::move(pols[i]), vals[i]});
      } catch (const std::exception& e) {
        reportErr(e.what());
        evalDead.store(true);
        fulfillFallback(batch);
        std::vector<EvalReq*> rest;
        {
          std::lock_guard<std::mutex> lk(qMtx);
          while (!queue.empty()) {
            rest.push_back(queue.front());
            queue.pop();
          }
          evalStop = true;
        }
        fulfillFallback(rest);
        requestStop();
        qCv.notify_all();
        return;
      } catch (...) {
        reportErr("unknown exception");
        evalDead.store(true);
        fulfillFallback(batch);
        std::vector<EvalReq*> rest;
        {
          std::lock_guard<std::mutex> lk(qMtx);
          while (!queue.empty()) {
            rest.push_back(queue.front());
            queue.pop();
          }
          evalStop = true;
        }
        fulfillFallback(rest);
        requestStop();
        qCv.notify_all();
        return;
      }
    }
  });

  auto submit = [&](const Graph* g, uint64_t key, const Position& pos) {
    // キャッシュは evaluateLeaves 内。ここはキューへ。
    EvalReq req;
    req.g = g;
    req.key = key;
    req.pos = pos;
    auto fut = req.prom.get_future();
    {
      std::lock_guard<std::mutex> lk(qMtx);
      if (evalDead.load()) {
        // 評価スレッドが既に退出: 自らフォールバック解決してデッドロックを防ぐ
        evalStop = true;
      } else {
        queue.push(&req);
      }
    }
    if (evalDead.load()) {
      std::vector<EvalReq*> one{&req};
      fulfillFallback(one);
      requestStop();
    } else {
      qCv.notify_one();
    }
    return fut.get();
  };

  std::atomic<int> launched{0};
  auto stop = [&] {
    if (stopFlag && stopFlag->load()) return true;
    {
      std::lock_guard<std::mutex> lk(treeMtx);
      if (root_->terminal) return true;
    }
    auto now = Clock::now();
    if (hasHard && now >= hardDeadline) return true;
    if (hasSoft && now >= softDeadline) {
      std::lock_guard<std::mutex> lk(treeMtx);
      if (isVisitStable(*root_)) return true;
    }
    return false;
  };

  auto backup = [](std::mutex& m, std::vector<PathStep>& path, float leafValue) {
    std::lock_guard<std::mutex> lk(m);
    float v = leafValue;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      v = -v;
      v = backupStep(it->node, it->child, v);
    }
  };

  // リーフ詰み探索の打ち切り条件 (並列探査経路): 時間制御があれば締切、
  // 無くてもノード上限で必ず降りる (タイマーだけでは止まらない暴走を防ぐ)。
  MateAbort mateAbort;
  mateAbort.stopFlag = stopFlag;
  mateAbort.deadlineTicks = hasHard ? hardDeadline.time_since_epoch().count() : 0;
  mateAbort.nodeBudget = kMateLeafNodeBudget;

  auto worker = [&] {
    while (true) {
      if (stop()) break;
      if (launched.fetch_add(1) >= params_.simulations) break;

      try {
        std::vector<PathStep> path;
        Position pos = rootPos;
        MctsNode* node;
        bool repetition = false;
        {
          std::lock_guard<std::mutex> lk(treeMtx);
          node = root_.get();
          std::vector<uint64_t> seen = history;
          seen.push_back(pos.hashKey());
          while (node->expanded && !node->terminal) {
            int c = node->selectChild(params_.cPuct, params_.fpuReduction);
            node->VL[c]++;
            path.push_back({node, c});
            pos.doMove(node->moves[c]);
            uint64_t key = pos.hashKey();
            if (int(std::count(seen.begin(), seen.end(), key)) + 1 >= 4) {
              repetition = true;
              break;
            }
            seen.push_back(key);
            if (!node->children[c]) node->children[c] = std::make_unique<MctsNode>();
            node = node->children[c].get();
          }
        }

        if (repetition) {
          backup(treeMtx, path, 0.f);
          continue;
        }
        if (node->terminal) {
          backup(treeMtx, path, node->terminalValue);
          continue;
        }

        Graph g = buildGraph(pos);
        if (g.moves.empty()) {
          {
            std::lock_guard<std::mutex> lk(treeMtx);
            node->terminal = true;
            node->terminalValue = -1.f;
            node->expanded = true;
          }
          backup(treeMtx, path, -1.f);
          continue;
        }
        if (params_.mateDepth > 0 &&
            canForceMate(pos, params_.mateDepth, nullptr, &mateAbort)) {
          {
            std::lock_guard<std::mutex> lk(treeMtx);
            node->terminal = true;
            node->terminalValue = +1.f;
            node->expanded = true;
          }
          backup(treeMtx, path, +1.f);
          continue;
        }

        auto [policy, value] = submit(&g, pos.hashKey(), pos);
        {
          std::lock_guard<std::mutex> lk(treeMtx);
          if (!node->expanded) {
            node->moves = g.moves;
            node->P = policy;
            size_t n = node->moves.size();
            node->N.assign(n, 0);
            node->W.assign(n, 0.f);
            node->VL.assign(n, 0);
            node->children.resize(n);
            node->expanded = true;
          }
          float v = value;
          for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;
            v = backupStep(it->node, it->child, v);
          }
        }
      } catch (const std::exception& e) {
        reportErr(std::string("worker: ") + e.what());
        requestStop();
        break;
      } catch (...) {
        reportErr("worker: unknown exception");
        requestStop();
        break;
      }
    }
  };

  std::vector<std::thread> pool;
  for (int t = 0; t < params_.threads; t++) pool.emplace_back(worker);
  for (auto& th : pool) th.join();

  {
    std::lock_guard<std::mutex> lk(qMtx);
    evalStop = true;
  }
  qCv.notify_all();
  evalThread.join();
}

int Search::totalVisits(const MctsNode& root) const {
  int n = 0;
  for (int v : root.N) n += v;
  return n;
}

Move Search::ponderMove(const MctsNode& root, Move best) const {
  for (size_t i = 0; i < root.moves.size(); i++) {
    if (root.moves[i] != best) continue;
    const MctsNode* child = root.children[i].get();
    if (!child || !child->expanded || child->moves.empty()) return MOVE_NONE;
    int bi = 0;
    for (size_t j = 1; j < child->N.size(); j++)
      if (child->N[j] > child->N[bi]) bi = int(j);
    return child->N[bi] > 0 ? child->moves[bi] : MOVE_NONE;
  }
  return MOVE_NONE;
}

std::vector<float> Search::visitDistribution(const MctsNode& root) const {
  std::vector<float> dist(root.N.size(), 0.f);
  float total = 0;
  for (int n : root.N) total += n;
  if (total <= 0) return dist;
  for (size_t i = 0; i < root.N.size(); i++) dist[i] = root.N[i] / total;
  return dist;
}

Move Search::selectMove(const MctsNode& root, float temperature) {
  if (root.moves.empty()) return MOVE_NONE;
  if (provenMateMove_ != MOVE_NONE) return provenMateMove_;
  for (size_t i = 0; i < root.moves.size(); i++) {
    const MctsNode* c = root.children[i].get();
    if (c && c->terminal && c->terminalValue <= -0.5f) return root.moves[i];
  }
  if (temperature <= 1e-6f) {
    int best = 0;
    for (size_t i = 1; i < root.N.size(); i++)
      if (root.N[i] > root.N[best]) best = int(i);
    return root.moves[best];
  }
  std::vector<double> w;
  for (int n : root.N) w.push_back(std::pow(double(n), 1.0 / temperature));
  std::discrete_distribution<int> dd(w.begin(), w.end());
  return root.moves[dd(rng_)];
}

} // namespace hisen
