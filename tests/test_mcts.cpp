#include "../src/mcts/search.h"
#include "../src/engine/mate.h"
#include "../src/engine/movegen.h"
#include <gtest/gtest.h>

using namespace hisen;

namespace {
SearchParams smallParams() {
  SearchParams sp;
  sp.simulations = 64;
  sp.batchSize = 8;
  sp.seed = 7;
  sp.threads = 1;
  sp.tacticalDepth = 1;
  sp.mateDepth = 3;
  sp.evalCacheSize = 10000;
  return sp;
}
} // namespace

TEST(Mcts, ReturnsLegalMove) {
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  Search search(eval, smallParams());
  Position pos = Position::startpos();
  MctsNode* root = search.run(pos, {});
  Move m = search.selectMove(*root, 0.f);
  EXPECT_TRUE(isLegal(pos, m));
  int total = 0;
  for (int n : root->N) total += n;
  EXPECT_GE(total, 60);
}

TEST(Mcts, MatePositionReturnsNone) {
  Position pos = Position::fromSfen("4k4/4G4/4P4/9/9/9/9/9/4K4 w - 1");
  ASSERT_TRUE(legalMoves(pos).empty());
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  Search search(eval, smallParams());
  MctsNode* root = search.run(pos, {});
  EXPECT_TRUE(root->terminal);
  EXPECT_EQ(search.selectMove(*root, 0.f), MOVE_NONE);
}

TEST(Mcts, FindsMateInOne) {
  Position pos = Position::fromSfen("4k4/9/4P4/9/9/9/9/9/4K4 b G 1");
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  SearchParams sp = smallParams();
  sp.simulations = 256;
  sp.batchSize = 16;
  Search search(eval, sp);
  MctsNode* root = search.run(pos, {});
  Move m = search.selectMove(*root, 0.f);
  EXPECT_EQ(usiMove(m), "G*5b");
}

TEST(Mcts, VisitDistributionNormalized) {
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  Search search(eval, smallParams());
  MctsNode* root = search.run(Position::startpos(), {});
  auto dist = search.visitDistribution(*root);
  float sum = 0;
  for (float p : dist) sum += p;
  EXPECT_NEAR(sum, 1.f, 1e-4);
}

TEST(Mcts, TreeReuseAfterMove) {
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  SearchParams sp = smallParams();
  sp.simulations = 128;
  Search search(eval, sp);

  Position pos = Position::startpos();
  MctsNode* root1 = search.run(pos, {});
  Move m = search.selectMove(*root1, 0.f);
  ASSERT_TRUE(isLegal(pos, m));
  int visits1 = search.totalVisits(*root1);
  ASSERT_GT(visits1, 0);

  // 最善手の子に降りた局面で再探索 → 再利用ヒット
  Position next = pos;
  next.doMove(m);
  std::vector<uint64_t> hist = {pos.hashKey()};
  MctsNode* root2 = search.run(next, hist);
  EXPECT_TRUE(search.lastReuseHit());
  EXPECT_TRUE(isLegal(next, search.selectMove(*root2, 0.f)));
}

TEST(Mcts, ClearTreeDisablesReuse) {
  HisenNet net(64, 2, 2, 4, 128);
  Evaluator eval(net, torch::Device(torch::kCPU));
  Search search(eval, smallParams());
  Position pos = Position::startpos();
  MctsNode* root = search.run(pos, {});
  Move m = search.selectMove(*root, 0.f);
  search.clearTree();
  Position next = pos;
  next.doMove(m);
  search.run(next, {pos.hashKey()});
  EXPECT_FALSE(search.lastReuseHit());
}

TEST(Mate, MateInOneWithTT) {
  clearMateTable();
  Position pos = Position::fromSfen("4k4/9/4P4/9/9/9/9/9/4K4 b G 1");
  Move m = MOVE_NONE;
  EXPECT_TRUE(canForceMate(pos, 1, &m));
  EXPECT_EQ(usiMove(m), "G*5b");
  // 2 回目は置換表経由でも true
  EXPECT_TRUE(canForceMate(pos, 1, nullptr));
}

TEST(Mate, NoMateOnStartpos) {
  clearMateTable();
  EXPECT_FALSE(canForceMate(Position::startpos(), 3, nullptr));
}
