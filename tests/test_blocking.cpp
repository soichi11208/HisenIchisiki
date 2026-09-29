#include "../src/engine/blocking.h"
#include <gtest/gtest.h>

using namespace hisen;

namespace {
bool hasEdge(const std::vector<BlockingEdge>& edges, Square src, Square dst) {
  for (auto& e : edges)
    if (e.src == src && e.dst == dst) return true;
  return false;
}
} // namespace

TEST(Blocking, PinByRook) {
  // 後手飛 5a — 先手金 5e — 先手玉 5i: 飛→玉 の遮蔽辺 (ピン)
  Position pos = Position::fromSfen("4r4/9/9/9/4G4/9/9/9/4K4 b - 1");
  auto edges = computeBlockingEdges(pos);
  Square rook = makeSquare(0, 4), gold = makeSquare(4, 4), king = makeSquare(8, 4);
  ASSERT_TRUE(hasEdge(edges, rook, king));
  for (auto& e : edges)
    if (e.src == rook && e.dst == king) EXPECT_EQ(e.blocker, gold);
}

TEST(Blocking, OnlyOnePieceSkipped) {
  // 飛の先に駒が 3 枚: 2 枚目までしか辺を張らない
  Position pos = Position::fromSfen("4r4/9/4P4/9/4P4/9/4P4/9/4K4 b - 1");
  auto edges = computeBlockingEdges(pos);
  Square rook = makeSquare(0, 4);
  EXPECT_TRUE(hasEdge(edges, rook, makeSquare(4, 4)));  // 1 枚越し
  EXPECT_FALSE(hasEdge(edges, rook, makeSquare(6, 4))); // 2 枚越しは張らない
  EXPECT_FALSE(hasEdge(edges, rook, makeSquare(8, 4)));
}

TEST(Blocking, BishopDiagonal) {
  // 角 9i — 歩 5e — 玉 1a 型の斜めピン
  Position pos = Position::fromSfen("8k/9/9/9/4P4/9/9/9/B8 w - 1");
  auto edges = computeBlockingEdges(pos);
  EXPECT_TRUE(hasEdge(edges, makeSquare(8, 0), makeSquare(0, 8)));
}

TEST(Blocking, NonSliderHasNoEdges) {
  Position pos = Position::fromSfen("4k4/9/9/9/4G4/4P4/9/9/4K4 b - 1");
  for (auto& e : computeBlockingEdges(pos)) {
    PieceType pt = typeOf(pos.board[e.src]);
    EXPECT_TRUE(isSlider(pt));
  }
}

TEST(Blocking, StartposCount) {
  // 初期局面: 香 4 本は歩 1 枚越しに何もない/盤端、飛角も同様に少数
  auto edges = computeBlockingEdges(Position::startpos());
  // 例: 先手飛 2h → 2g 歩越しに 2c 歩 (後手) への遮蔽辺がある
  Square rook = makeSquare(7, 7), target = makeSquare(2, 7);
  EXPECT_TRUE(hasEdge(edges, rook, target));
}
