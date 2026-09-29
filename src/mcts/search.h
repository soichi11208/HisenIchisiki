#pragma once
#include "../model/hisen.h"
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace hisen {

// NN 評価器: 局面集合をバッチで forward し、合法手分布と value を返す。
struct Evaluator {
  HisenNet net;
  torch::Device device;
  bool amp; // bf16 autocast 推論 (CUDA のみ有効、学習と同じ精度で高速化)

  Evaluator(HisenNet n, torch::Device dev, bool useAmp = true)
      : net(std::move(n)), device(dev), amp(useAmp) {
    net->to(device);
    net->eval();
  }

  struct Result {
    std::vector<std::vector<float>> policies; // 各局面の合法手確率
    std::vector<float> values;                // 手番側視点 [-1,1]
  };
  virtual Result evaluate(const std::vector<const Graph*>& graphs);
  virtual ~Evaluator() = default;
};

// 複数スレッドの評価リクエストを 1 つの大きなバッチにまとめて GPU 推論するラッパ。
class BatchedEvaluator : public Evaluator {
 public:
  BatchedEvaluator(HisenNet net, torch::Device dev, int maxBatch, bool useAmp = true);
  ~BatchedEvaluator() override;
  Result evaluate(const std::vector<const Graph*>& graphs) override;

 private:
  struct Req {
    std::vector<const Graph*> graphs;
    std::promise<Result> prom;
  };
  void workerLoop();
  int maxBatch_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<Req*> queue_;
  bool stop_ = false;
  std::thread worker_;
};

struct MctsNode {
  std::vector<Move> moves;
  std::vector<float> P;
  std::vector<int> N;
  std::vector<float> W;
  std::vector<int> VL; // virtual loss
  std::vector<std::unique_ptr<MctsNode>> children;
  bool expanded = false;
  bool terminal = false;
  float terminalValue = 0.f; // 手番側視点

  int totalN() const;
  // FPU (First Play Urgency) Reduction: 未訪問の子の Q を「親の Q - fpuReduction」で見積もる。
  int selectChild(float cPuct, float fpuReduction) const;
};

struct SearchParams {
  int simulations = 800;
  int batchSize = 32;     // 1 回の NN forward に詰める最大局面数
  int threads = 1;        // >1 で並列 MCTS (ワーカー群 + 集約評価スレッド)
  int mateDepth = 5;      // ルート/リーフの詰み探索手数 (0 で無効)
  int tacticalDepth = 0;  // リーフの戦術 qsearch 深さ (0 で無効、既定)。>0 は CPU律速で nps 半減
  size_t evalCacheSize = 250000; // 評価キャッシュ最大エントリ (0 で無効)
  float cPuct = 1.5f;
  float fpuReduction = 0.25f;
  float dirichletAlpha = 0.15f;
  float dirichletEps = 0.25f; // 0 でノイズ無効
  // 時間制御: soft 以降は着手安定なら打ち切り、hard で強制終了。
  // soft 未設定 (default) なら hard のみ (従来どおり)。
  int minVisitsForStop = 200;       // 安定打ち切りに必要な最少訪問
  float stableVisitRatio = 2.0f;    // 1位訪問 >= 2位 * この倍率 なら安定
  uint64_t seed = 0;                // 0 なら random_device
};

class Search {
 public:
  Search(Evaluator& eval, SearchParams params);

  // root から params.simulations 回探索し root を返す (千日手は引き分け扱い)。
  // 前回の探索木が new root へ辿れるなら再利用する。
  // history: 千日手検出用の経路ハッシュ (現局面は含めない想定)。
  // hardDeadline: 過ぎたら強制打ち切り。softDeadline: 過ぎかつ着手安定なら早期終了。
  using Clock = std::chrono::steady_clock;
  MctsNode* run(const Position& root, const std::vector<uint64_t>& history,
                const std::atomic<bool>* stopFlag = nullptr,
                Clock::time_point hardDeadline = Clock::time_point{},
                Clock::time_point softDeadline = Clock::time_point{});

  void setSimulations(int n) { params_.simulations = n; }
  void setParams(const SearchParams& p) { params_ = p; }
  const SearchParams& params() const { return params_; }

  // ツリー・評価キャッシュを破棄 (usinewgame / モデル再読込時)。
  void clearTree();

  int totalVisits(const MctsNode& root) const;
  // 1位・2位の訪問回数 (時間制御の安定判定用)。無ければ 0。
  void topTwoVisits(const MctsNode& root, int& best, int& second) const;
  bool isVisitStable(const MctsNode& root) const;

  // best を指した後の相手の最有力応手 (ponder 用)。
  Move ponderMove(const MctsNode& root, Move best) const;

  Move selectMove(const MctsNode& root, float temperature);
  std::vector<float> visitDistribution(const MctsNode& root) const;

  // 直前探索でツリー再利用に成功したか (info string 用)。
  bool lastReuseHit() const { return lastReuseHit_; }

 private:
  Evaluator& eval_;
  SearchParams params_;
  std::mt19937_64 rng_;
  std::unique_ptr<MctsNode> root_;
  Position rootPos_{};
  bool haveTree_ = false;
  bool lastReuseHit_ = false;
  Move provenMateMove_ = MOVE_NONE;

  // --- 評価キャッシュ (hash → policy/value)。並列時は mutex 保護。---
  struct CacheEntry {
    std::vector<Move> moves;
    std::vector<float> policy;
    float value = 0.f;
  };
  mutable std::mutex cacheMtx_;
  std::unordered_map<uint64_t, CacheEntry> evalCache_;

  void addDirichletNoise(MctsNode& node);
  void clearVirtualLoss(MctsNode* node);
  bool tryReuseRoot(const Position& target);
  bool findPathTo(MctsNode* node, Position pos, const Position& target,
                  std::vector<int>& path, int maxDepth) const;

  bool cacheLookup(uint64_t key, const std::vector<Move>& moves,
                   std::vector<float>& policy, float& value) const;
  void cacheStore(uint64_t key, const std::vector<Move>& moves,
                  const std::vector<float>& policy, float value);

  // リーフ戦術 qsearch (捕獲・王手のみ)。nnValue を stand-pat に使う。
  float leafTactical(const Position& pos, float nnValue) const;

  // NN 評価 (+ キャッシュ + 戦術補正)。graphs と keys/pos は同順。
  void evaluateLeaves(const std::vector<const Graph*>& graphs,
                      const std::vector<uint64_t>& keys,
                      const std::vector<Position>& positions,
                      std::vector<std::vector<float>>& policies,
                      std::vector<float>& values);

  void runParallel(const Position& rootPos, const std::vector<uint64_t>& history,
                   const std::atomic<bool>* stopFlag, Clock::time_point hardDeadline,
                   Clock::time_point softDeadline, bool hasHard, bool hasSoft);
};

} // namespace hisen
