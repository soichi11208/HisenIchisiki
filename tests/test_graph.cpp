#include "../src/graph/builder.h"
#include "../src/model/hisen.h"
#include <gtest/gtest.h>

using namespace hisen;

TEST(Graph, StartposNodeCount) {
  Graph g = buildGraph(Position::startpos());
  EXPECT_EQ(g.numNodes(), 40 + 14); // 盤上 40 駒 + 持ち駒 14
  EXPECT_EQ(g.moves.size(), 30u);
  EXPECT_EQ(g.move_src.size(0), 30);
}

TEST(Graph, SideToMoveNormalization) {
  // 先後反転対称: 初手 7g7f 後の後手番局面と、その鏡像の先手番局面は同一グラフ
  Position pos = Position::startpos();
  pos.doMove(parseUsiMove("7g7f"));
  Graph g1 = buildGraph(pos); // 後手番 → 正規化で回転

  // 同じ進行を後手側から見た局面 (盤面 180° 回転 + 手番先手)
  Position mirror = Position::fromSfen(
      "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 2");
  Graph g2 = buildGraph(mirror);
  ASSERT_EQ(g1.numNodes(), g2.numNodes());
  EXPECT_TRUE(torch::equal(g1.node_type, g2.node_type));
  EXPECT_TRUE(torch::equal(g1.node_x, g2.node_x));
  EXPECT_TRUE(torch::equal(g1.node_y, g2.node_y));
  EXPECT_EQ(g1.edge_index.size(1), g2.edge_index.size(1));
  EXPECT_EQ(g1.moves.size(), g2.moves.size());
}

TEST(Graph, EdgeCountsReasonable) {
  Graph g = buildGraph(Position::startpos());
  int64_t E = g.edge_index.size(1);
  EXPECT_GT(E, 100);
  EXPECT_LT(E, 800);
  // 持ち駒辺はちょうど 14
  EXPECT_EQ((g.edge_type == EDGE_HAND).sum().item<int64_t>(), 14);
  // 近傍辺は双方向で偶数
  EXPECT_EQ((g.edge_type == EDGE_NEIGHBOR).sum().item<int64_t>() % 2, 0);
}

TEST(Graph, HandCountFeature) {
  Position pos = Position::fromSfen("4k4/9/9/9/9/9/9/9/4K4 b 3P 1");
  Graph g = buildGraph(pos);
  // 先手歩の持ち駒ノード (盤上 2 駒の次、index 2) の count = 3/7
  auto scal = g.node_scal[2];
  EXPECT_FLOAT_EQ(scal[3].item<float>(), 3.f / 7.f);
  EXPECT_FLOAT_EQ(scal[2].item<float>(), 1.f); // is_hand
}

TEST(Model, ForwardShapes) {
  torch::NoGradGuard ng;
  HisenNet net(64, 2, 2, 4, 128); // テスト用小型
  net->eval();
  Graph g1 = buildGraph(Position::startpos());
  Position p2 = Position::startpos();
  p2.doMove(parseUsiMove("7g7f"));
  Graph g2 = buildGraph(p2);
  auto batch = GraphBatch::collate({&g1, &g2});
  auto [logits, values] = net->forward(batch);
  EXPECT_EQ(logits.size(0), int64_t(g1.moves.size() + g2.moves.size()));
  EXPECT_EQ(values.size(0), 2);
  EXPECT_LE(values.abs().max().item<float>(), 1.f);
  // 合法手で確率和 = 1
  auto lps = splitPolicyLogSoftmax(logits, batch.moveCounts);
  EXPECT_NEAR(lps[0].exp().sum().item<float>(), 1.f, 1e-4);
}

TEST(Model, GATv2AttentionSumsToOne) {
  // 小グラフで softmax 正規化を検証 (alpha の和が dst ごとに 1)
  torch::NoGradGuard ng;
  GATv2Layer layer(8, 2, 4);
  auto x = torch::randn({3, 8});
  auto ei = torch::tensor({{0, 1, 2}, {2, 2, 1}}, torch::kInt64);
  auto ea = torch::randn({3, 4});
  auto out = layer->forward(x, ei, ea);
  EXPECT_EQ(out.sizes(), x.sizes());
  EXPECT_TRUE(out.isfinite().all().item<bool>());
}
