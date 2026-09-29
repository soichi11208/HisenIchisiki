// 緋閃壱式 USI エンジン
#include "../engine/movegen.h"
#include "../mcts/search.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <unordered_map>

using namespace hisen;

namespace {

// 思考時間から差し引く通信・推論オーバーヘッドの安全マージン (ms)
constexpr int64_t kOverheadMs = 80;
// go ponder に時間情報が無い場合の ponderhit 打ち切り保険 (ms)。
// 時間情報が無いと hardMs=0 になりタイマーが起動せず bestmove が永遠に出ない。
constexpr int64_t kPonderHitFallbackMs = 5000;

std::string exeDir() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = '\0';
  return std::filesystem::path(buf).parent_path().string();
}

std::string resolveFile(const std::string& configured) {
  if (configured.empty()) return configured;
  if (std::filesystem::exists(configured)) return configured;
  std::string ed = exeDir();
  std::string base = std::filesystem::path(configured).filename().string();
  for (const std::string& c : {ed + "/" + configured,
                               ed + "/weights/" + base,
                               ed + "/../weights/" + base})
    if (std::filesystem::exists(c)) return c;
  return configured;
}

struct GoLimits {
  int64_t btime = 0, wtime = 0, binc = 0, winc = 0, byoyomi = 0, movetime = 0;
  int nodes = 0;
  bool infinite = false, ponder = false;
};

// soft = 安定なら切ってよい時間、hard = 強制打ち切り
struct TimeBudget {
  int64_t softMs = 0; // 0 = 時間制限なし
  int64_t hardMs = 0;
  int64_t minMs = 0;  // これ未満では安定打ち切りしない (run 側は soft で制御)
};

struct EngineState {
  std::string modelPath = "weights/hisen_latest.pt";
  std::string bookPath = "weights/book.bin";
  int nodes = 800;
  int batchSize = 64;
  int threads = 32;
  int mateDepth = 5;
  int tacticalDepth = 0;
  int gpuId = 0;
  int bookMoves = 40;
  float fpuReduction = 0.25f;
  bool ponderEnabled = true;
  bool dirty = true; // true なら次回 isready で再初期化が必要
  Position pos = Position::startpos();
  std::vector<uint64_t> history;
  std::unique_ptr<Evaluator> eval;
  std::unique_ptr<Search> search;
  std::atomic<bool> stopFlag{false};
  std::thread searchThread;
  // ponder 中の bestmove は USI 仕様上 stop/ponderhit まで出さない。
  // 早期終了 (ルート詰み/強制詰み発見) した分は保留し、指示が来たら出力する。
  std::mutex outMtx;
  bool ponderActive = false;
  bool hasPendingBest = false;
  std::string pendingBest;
  std::unordered_map<uint64_t, std::vector<std::pair<Move, uint32_t>>> book;
  std::mutex timerMtx;
  std::condition_variable timerCv;
  bool timerCancel = false;
  std::thread timerThread;
  GoLimits lastLimits;
};

// bestmove 出力。ponder 実行中は出力せず保留する (USI: stop/ponderhit まで出さない)。
void emitOrStash(EngineState& st, const std::string& line) {
  std::lock_guard<std::mutex> lk(st.outMtx);
  if (st.ponderActive) {
    st.pendingBest = line;
    st.hasPendingBest = true;
  } else {
    std::cout << line << std::flush;
  }
}

// 保留中の bestmove を破棄する (新局面の開始など、もう不要な場合)。
void dropPendingBest(EngineState& st) {
  std::lock_guard<std::mutex> lk(st.outMtx);
  st.hasPendingBest = false;
  st.pendingBest.clear();
}

// stop/ponderhit 時: ponder を終了扱いにし、保留があれば出力する。出力したら true。
bool flushPendingBest(EngineState& st) {
  std::string out;
  {
    std::lock_guard<std::mutex> lk(st.outMtx);
    st.ponderActive = false;
    if (st.hasPendingBest) {
      out.swap(st.pendingBest);
      st.hasPendingBest = false;
    }
  }
  if (out.empty()) return false;
  std::cout << out << std::flush;
  return true;
}

void loadBook(EngineState& st) {
  st.book.clear();
  st.book.clear();
  std::string bp = resolveFile(st.bookPath);
  FILE* f = std::fopen(bp.c_str(), "rb");
  if (!f) {
    std::cout << "info string no book (searched " << bp << ")" << std::endl;
    return;
  }
  uint64_t h;
  uint32_t m, c;
  size_t n = 0;
  while (std::fread(&h, 8, 1, f) == 1 && std::fread(&m, 4, 1, f) == 1 &&
         std::fread(&c, 4, 1, f) == 1) {
    st.book[h].push_back({Move(m), c});
    n++;
  }
  std::fclose(f);
  std::cout << "info string book loaded: " << n << " entries from " << bp << std::endl;
}

Move probeBook(EngineState& st) {
  if (st.bookMoves <= 0 || st.pos.ply >= st.bookMoves || st.book.empty()) return MOVE_NONE;
  auto it = st.book.find(st.pos.hashKey());
  if (it == st.book.end()) return MOVE_NONE;
  Move best = MOVE_NONE;
  uint32_t bestCount = 0;
  for (auto& [m, c] : it->second)
    if (c > bestCount && isLegal(st.pos, m)) {
      best = m;
      bestCount = c;
    }
  return best;
}

void setPosition(EngineState& st, std::istringstream& ss) {
  std::string tok;
  ss >> tok;
  st.history.clear();
  if (tok == "startpos") {
    st.pos = Position::startpos();
  } else if (tok == "sfen") {
    std::string b, s, h, p;
    ss >> b >> s >> h >> p;
    st.pos = Position::fromSfen(b + " " + s + " " + h + " " + p);
  }
  ss >> tok; // "moves"
  st.history.push_back(st.pos.hashKey());
  std::string mv;
  while (ss >> mv) {
    st.pos.doMove(parseUsiMove(mv));
    st.history.push_back(st.pos.hashKey());
  }
  st.history.pop_back(); // run() 側で現局面を積むため除去
}

void cancelTimer(EngineState& st) {
  {
    std::lock_guard<std::mutex> lk(st.timerMtx);
    st.timerCancel = true;
  }
  st.timerCv.notify_all();
  if (st.timerThread.joinable()) st.timerThread.join();
  st.timerCancel = false;
}

void startTimer(EngineState& st, int64_t ms) {
  cancelTimer(st);
  st.timerThread = std::thread([&st, ms] {
    std::unique_lock<std::mutex> lk(st.timerMtx);
    if (!st.timerCv.wait_for(lk, std::chrono::milliseconds(ms),
                             [&] { return st.timerCancel; }))
      st.stopFlag = true;
  });
}

void joinSearch(EngineState& st) {
  cancelTimer(st);
  st.stopFlag = true;
  if (st.searchThread.joinable()) st.searchThread.join();
}

GoLimits parseGo(std::istringstream& ss) {
  GoLimits g;
  std::string t;
  while (ss >> t) {
    if (t == "btime") ss >> g.btime;
    else if (t == "wtime") ss >> g.wtime;
    else if (t == "binc") ss >> g.binc;
    else if (t == "winc") ss >> g.winc;
    else if (t == "byoyomi") ss >> g.byoyomi;
    else if (t == "movetime") ss >> g.movetime;
    else if (t == "nodes") ss >> g.nodes;
    else if (t == "infinite") g.infinite = true;
    else if (t == "ponder") g.ponder = true;
  }
  return g;
}

// 持ち時間 → soft/hard 予算。
// soft: 残り/25 + 加算*0.8 + 秒読み*0.9。着手が安定していればここで切る。
// hard: soft の最大 3 倍、または残り+秒読みの安全枠。不安定なら延長。
TimeBudget computeBudget(const GoLimits& g, Color side) {
  TimeBudget b;
  if (g.infinite || g.ponder) return b;
  if (g.movetime > 0) {
    b.hardMs = std::max<int64_t>(g.movetime - kOverheadMs, 10);
    b.softMs = std::max<int64_t>(b.hardMs * 7 / 10, 10);
    b.minMs = std::min<int64_t>(b.softMs / 3, 200);
    return b;
  }
  int64_t remain = side == BLACK ? g.btime : g.wtime;
  int64_t inc = side == BLACK ? g.binc : g.winc;
  if (remain == 0 && g.byoyomi == 0 && inc == 0) return b; // ノード制

  int64_t soft = remain / 25 + inc * 8 / 10 + g.byoyomi * 9 / 10;
  // 終盤寄り (残りが少ない) はもう少し厚く
  if (remain > 0 && remain < 30000) soft = remain / 15 + inc * 8 / 10 + g.byoyomi * 9 / 10;
  int64_t cap = remain + g.byoyomi - kOverheadMs;
  if (cap < 10) cap = 10;
  soft = std::max<int64_t>(10, std::min(soft, cap));

  int64_t hard = std::min(cap, std::max(soft + soft / 2, soft * 3));
  // 秒読み主体なら hard も秒読み枠内に抑える
  if (g.byoyomi > 0 && remain < g.byoyomi) {
    hard = std::min(hard, g.byoyomi - kOverheadMs);
    soft = std::min(soft, hard);
  }
  b.softMs = soft;
  b.hardMs = std::max(soft, hard);
  b.minMs = std::min<int64_t>(soft / 4, 300);
  return b;
}

void go(EngineState& st, const GoLimits& lim) {
  joinSearch(st);
  dropPendingBest(st); // 新しい探索を始めるので前回の保留は破棄
  st.stopFlag = false;
  st.lastLimits = lim;
  {
    std::lock_guard<std::mutex> lk(st.outMtx);
    st.ponderActive = lim.ponder; // ponder 中は bestmove を保留する
  }

  if (!st.search) {
    emitOrStash(st, "info string ERROR search not ready; resigning\nbestmove resign\n");
    return;
  }

  if (!lim.ponder && !lim.infinite) {
    Move bm = probeBook(st);
    if (bm != MOVE_NONE) {
      // ブック手ではツリーが局面と食い違うので破棄
      st.search->clearTree();
      std::cout << "info string book move" << std::endl;
      emitOrStash(st, "bestmove " + usiMove(bm) + "\n");
      return;
    }
  }

  TimeBudget budget = computeBudget(lim, st.pos.side);
  int simCap = lim.nodes > 0                ? lim.nodes
               : lim.infinite || lim.ponder ? 1000000000
               : budget.hardMs > 0          ? 1000000
                                            : st.nodes;
  st.search->setSimulations(simCap);

  // 時間制御パラメータを Search に反映 (min visits は予算に応じて緩和)
  {
    SearchParams sp = st.search->params();
    sp.mateDepth = st.mateDepth;
    sp.tacticalDepth = st.tacticalDepth;
    sp.fpuReduction = st.fpuReduction;
    if (budget.softMs > 0) {
      // 短考では早めに安定判定できるよう閾値を下げる
      sp.minVisitsForStop = budget.softMs < 1000 ? 80 : (budget.softMs < 5000 ? 150 : 200);
    }
    st.search->setParams(sp);
  }

  st.searchThread = std::thread([&st, budget] {
    try {
      auto start = std::chrono::steady_clock::now();
      Search::Clock::time_point hard{}, soft{};
      if (budget.hardMs > 0) {
        hard = start + std::chrono::milliseconds(budget.hardMs);
        // minMs 経過前は soft を遅らせ、最低思考時間を確保
        int64_t softMs = std::max(budget.softMs, budget.minMs);
        soft = start + std::chrono::milliseconds(softMs);
      }

      MctsNode* root = st.search->run(st.pos, st.history, &st.stopFlag, hard, soft);
      if (root->moves.empty()) {
        emitOrStash(st, "bestmove resign\n");
        return;
      }
      Move best = st.search->selectMove(*root, 0.f);
      int bestIdx = 0;
      for (size_t i = 0; i < root->moves.size(); i++)
        if (root->moves[i] == best) bestIdx = int(i);
      float q = root->N[bestIdx] > 0 ? root->W[bestIdx] / root->N[bestIdx] : 0.f;
      int visits = st.search->totalVisits(*root);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
      int64_t nps = ms > 0 ? int64_t(visits) * 1000 / ms : 0;
      Move pm = st.search->ponderMove(*root, best);
      int n1 = 0, n2 = 0;
      st.search->topTwoVisits(*root, n1, n2);
      std::cout << "info time " << ms << " nodes " << visits << " nps " << nps
                << " score cp " << int(q * 600) << " pv " << usiMove(best);
      if (pm != MOVE_NONE) std::cout << " " << usiMove(pm);
      std::cout << std::endl;
      if (st.search->lastReuseHit())
        std::cout << "info string tree reuse hit visits=" << visits
                  << " top=" << n1 << "/" << n2 << std::endl;
      std::string bmLine = "bestmove " + usiMove(best);
      if (st.ponderEnabled && pm != MOVE_NONE) bmLine += " ponder " + usiMove(pm);
      bmLine += "\n";
      emitOrStash(st, bmLine);
    } catch (const std::exception& e) {
      // 例外でプロセスが死ぬと bestmove 無し = 時間切れ。resign で応答して生存確保。
      std::cout << "info string ERROR search exception: " << e.what()
                << " -- resigning" << std::endl;
      emitOrStash(st, "bestmove resign\n");
    } catch (...) {
      std::cout << "info string ERROR search exception: unknown -- resigning"
                << std::endl;
      emitOrStash(st, "bestmove resign\n");
    }
  });
}

} // namespace

int main() {
  std::ios::sync_with_stdio(false);
  EngineState st;
  std::string line;

  while (std::getline(std::cin, line)) {
    std::istringstream ss(line);
    std::string cmd;
    ss >> cmd;
    try {
      if (cmd == "usi") {
        std::cout << "id name HisenIchishiki\n"
                  << "id author soichi\n"
                  << "option name ModelPath type string default weights/hisen_latest.pt\n"
                  << "option name Nodes type spin default 800 min 1 max 10000000\n"
                  << "option name BatchSize type spin default 64 min 1 max 1024\n"
                  << "option name Threads type spin default 32 min 1 max 128\n"
                  << "option name MateDepth type spin default 5 min 0 max 11\n"
                  << "option name TacticalDepth type spin default 0 min 0 max 4\n"
                  << "option name FpuReduction type string default 0.25\n"
                  << "option name GPUId type spin default 0 min -1 max 7\n"
                  << "option name BookFile type string default ./weights/book.bin\n"
                  << "option name BookMoves type spin default 40 min 0 max 100\n"
                  << "option name USI_Ponder type check default true\n"
                  << "usiok" << std::endl;
      } else if (cmd == "setoption") {
        std::string tok, name;
        ss >> tok >> name >> tok;
        std::string value;
        std::getline(ss, value);
        size_t p = value.find_first_not_of(" \t");
        value = p == std::string::npos ? "" : value.substr(p);
        auto safeInt = [&](int def) {
          try {
            return std::stoi(value);
          } catch (...) {
            std::cout << "info string WARN bad int for " << name << ": '" << value
                      << "' -> keep " << def << std::endl;
            return def;
          }
        };
        auto safeFloat = [&](float def) {
          try {
            return std::stof(value);
          } catch (...) {
            std::cout << "info string WARN bad float for " << name << ": '" << value
                      << "' -> keep " << def << std::endl;
            return def;
          }
        };
        if (name == "ModelPath") st.modelPath = value;
        else if (name == "Nodes") st.nodes = safeInt(st.nodes);
        else if (name == "BatchSize") st.batchSize = safeInt(st.batchSize);
        else if (name == "Threads") st.threads = safeInt(st.threads);
        else if (name == "MateDepth") st.mateDepth = safeInt(st.mateDepth);
        else if (name == "TacticalDepth") st.tacticalDepth = safeInt(st.tacticalDepth);
        else if (name == "FpuReduction") st.fpuReduction = safeFloat(st.fpuReduction);
        else if (name == "GPUId") st.gpuId = safeInt(st.gpuId);
        else if (name == "BookFile") st.bookPath = value;
        else if (name == "BookMoves") st.bookMoves = safeInt(st.bookMoves);
        else if (name == "USI_Ponder") st.ponderEnabled = (value == "true");
        if (name != "USI_Ponder") st.dirty = true; // 次回 isready で反映
      } else if (cmd == "isready") {
        if (st.search && !st.dirty) {
          std::cout << "readyok" << std::endl;
        } else {
          joinSearch(st); // 走っている (残留 ponder を含む) 思考を先に停止
          try {
            loadBook(st);
            std::string mp = resolveFile(st.modelPath);
            if (!std::filesystem::exists(mp)) {
              std::cout << "info string ERROR model NOT FOUND: '" << mp
                        << "' (searched as given, exe-relative, exe/weights/). "
                        << "Pass the model via USI: setoption name ModelPath value <path>"
                        << std::endl;
            } else {
              HisenNet net;
              torch::load(net, mp);
              std::cout << "info string loaded model: " << mp << std::endl;
              torch::Device dev = (st.gpuId >= 0 && torch::cuda::is_available())
                                      ? torch::Device(torch::kCUDA, st.gpuId)
                                      : torch::Device(torch::kCPU);
              st.eval = std::make_unique<Evaluator>(net, dev);
              SearchParams sp;
              sp.simulations = st.nodes;
              sp.batchSize = st.batchSize;
              sp.threads = st.threads;
              sp.mateDepth = st.mateDepth;
              sp.tacticalDepth = st.tacticalDepth;
              sp.fpuReduction = st.fpuReduction;
              sp.dirichletEps = 0.f;
              st.search = std::make_unique<Search>(*st.eval, sp);
              Graph g = buildGraph(st.pos);
              st.eval->evaluate({&g});
              std::cout << "info string ready (device=" << dev << " threads=" << st.threads
                        << " batch=" << st.batchSize << " mate=" << st.mateDepth
                        << " tactical=" << st.tacticalDepth << ")" << std::endl;
              st.dirty = false;
            }
          } catch (const std::exception& e) {
            // 失敗時は dirty のままにして次回 isready で再試行
            std::cout << "info string ERROR isready: " << e.what() << std::endl;
          } catch (...) {
            std::cout << "info string ERROR isready: unknown exception" << std::endl;
          }
          std::cout << "readyok" << std::endl;
        }
      } else if (cmd == "usinewgame") {
        // 思考中 (残留 ponder) の木を破棄すると use-after-free。先に停止。
        joinSearch(st);
        dropPendingBest(st);
        st.pos = Position::startpos();
        st.history.clear();
        if (st.search) st.search->clearTree();
      } else if (cmd == "position") {
        // 走っている検索が参照中の st.pos/st.history を書き換えないよう先に停止。
        joinSearch(st);
        dropPendingBest(st);
        setPosition(st, ss);
      } else if (cmd == "go") {
        go(st, parseGo(ss));
      } else if (cmd == "ponderhit") {
        // ponder が早期終了していれば保留分を出力。まだ走っていれば時間制限を課す。
        if (!flushPendingBest(st)) {
          GoLimits lim = st.lastLimits;
          lim.ponder = false;
          TimeBudget budget = computeBudget(lim, st.pos.side);
          // 時間情報が無い go ponder では hardMs=0 になる。保険を入れて必ず打ち切る。
          startTimer(st, budget.hardMs > 0 ? budget.hardMs : kPonderHitFallbackMs);
        }
      } else if (cmd == "stop") {
        joinSearch(st);
        flushPendingBest(st);
      } else if (cmd == "quit") {
        joinSearch(st);
        dropPendingBest(st);
        break;
      } else if (cmd == "gameover") {
        joinSearch(st);
        dropPendingBest(st);
        if (st.search) st.search->clearTree();
      }
    } catch (const std::exception& e) {
      std::cout << "info string ERROR cmd '" << cmd << "': " << e.what() << std::endl;
      if (cmd == "go") emitOrStash(st, "bestmove resign\n");
    } catch (...) {
      std::cout << "info string ERROR cmd '" << cmd << "': unknown exception"
                << std::endl;
      if (cmd == "go") emitOrStash(st, "bestmove resign\n");
    }
  }
  return 0;
}
