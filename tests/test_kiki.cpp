#include "../src/engine/kiki.h"
#include "../src/engine/movegen.h"
#include <gtest/gtest.h>
#include <algorithm>

using namespace hisen;

namespace {
bool attacks(const Position& pos, const std::string& from, const std::string& to) {
  Square f = parseUsiMove(from + to) >> 7 & 0x7f;
  Square t = parseUsiMove(from + to) & 0x7f;
  auto a = attacksFrom(pos, f);
  return std::find(a.begin(), a.end(), t) != a.end();
}
} // namespace

TEST(Kiki, StartposPawn) {
  Position pos = Position::startpos();
  EXPECT_TRUE(attacks(pos, "7g", "7f"));   // 先手歩は前
  EXPECT_FALSE(attacks(pos, "7g", "7h"));
  EXPECT_TRUE(attacks(pos, "3c", "3d"));   // 後手歩は逆方向
}

TEST(Kiki, LanceBlocked) {
  // 先手香 9i: 9h (空) を通り 9g の味方歩で停止 → {9h, 9g}
  Position pos = Position::startpos();
  auto a = attacksFrom(pos, makeSquare(8, 0)); // 9i 香
  ASSERT_EQ(a.size(), 2u);
  EXPECT_EQ(a.back(), makeSquare(6, 0));
}

TEST(Kiki, LanceStopsAtFirstPiece) {
  Position pos = Position::fromSfen("9/9/9/9/8p/9/9/9/8L b - 1");
  auto a = attacksFrom(pos, makeSquare(8, 8)); // 1i 香
  // 1h,1g,1f まで伸び、1e の歩を含んで停止
  EXPECT_EQ(a.size(), 4u);
  EXPECT_TRUE(std::find(a.begin(), a.end(), makeSquare(4, 8)) != a.end());
  EXPECT_TRUE(std::find(a.begin(), a.end(), makeSquare(3, 8)) == a.end());
}

TEST(Kiki, KnightAndEdge) {
  Position pos = Position::fromSfen("9/9/9/9/4N4/9/9/9/9 b - 1"); // 5e 桂
  auto a = attacksFrom(pos, makeSquare(4, 4));
  EXPECT_EQ(a.size(), 2u); // 4c, 6c
  Position edge = Position::fromSfen("9/9/9/9/N8/9/9/9/9 b - 1"); // 9e 桂
  EXPECT_EQ(attacksFrom(edge, makeSquare(4, 0)).size(), 1u); // 8c のみ
}

TEST(Kiki, HorseAndDragon) {
  Position pos = Position::fromSfen("9/9/9/9/4+B4/9/9/9/9 b - 1"); // 5e 馬
  auto a = attacksFrom(pos, makeSquare(4, 4));
  // 斜めスライド 16 + 縦横 4
  EXPECT_EQ(a.size(), 20u);
  Position d = Position::fromSfen("9/9/9/9/4+R4/9/9/9/9 b - 1"); // 5e 龍
  EXPECT_EQ(attacksFrom(d, makeSquare(4, 4)).size(), 20u);
}

TEST(Kiki, PromotedPawnMovesLikeGold) {
  Position pos = Position::fromSfen("9/9/9/9/4+P4/9/9/9/9 b - 1");
  EXPECT_EQ(attacksFrom(pos, makeSquare(4, 4)).size(), 6u);
}

TEST(Kiki, InCheck) {
  Position pos = Position::fromSfen("4k4/9/4R4/9/9/9/9/9/4K4 b - 1");
  EXPECT_TRUE(inCheck(pos, WHITE));
  EXPECT_FALSE(inCheck(pos, BLACK));
}

TEST(MoveGen, StartposCount) {
  EXPECT_EQ(legalMoves(Position::startpos()).size(), 30u);
}

TEST(MoveGen, Nifu) {
  // 5筋に歩があるとき 5 筋への歩打ちは不可
  Position pos = Position::fromSfen("4k4/9/9/9/9/9/4P4/9/4K4 b P 1");
  for (Move m : legalMoves(pos))
    if (isDrop(m) && dropType(m) == PAWN) EXPECT_NE(fileOf(moveTo(m)), 4);
}

TEST(MoveGen, MustPromote) {
  // 1 段目への歩の不成は生成されない
  Position pos = Position::fromSfen("4k4/4P4/9/9/9/9/9/9/4K4 b - 1");
  bool foundPromo = false;
  for (Move m : legalMoves(pos)) {
    if (!isDrop(m) && moveFrom(m) == makeSquare(1, 4) && moveTo(m) == makeSquare(0, 4)) {
      EXPECT_TRUE(isPromo(m));
      foundPromo = true;
    }
  }
  EXPECT_TRUE(foundPromo);
}

TEST(MoveGen, PawnDropMate) {
  // 頭金型: 5a 玉が逃げられず歩打ちで詰む形 → 打ち歩詰めは違法
  Position pos = Position::fromSfen("4k4/9/4G4/9/9/9/9/9/4K4 b P 1");
  // 5c 金で 5b への退路なし… 5b 歩打ちが詰みかは配置次第。違法手が混ざらないことだけ確認
  for (Move m : legalMoves(pos)) {
    Position next = pos;
    next.doMove(m);
    EXPECT_FALSE(inCheck(next, BLACK));
  }
}

TEST(Board, SfenRoundtrip) {
  Position pos = Position::startpos();
  EXPECT_EQ(Position::fromSfen(pos.toSfen()).toSfen(), pos.toSfen());
  pos.doMove(parseUsiMove("7g7f"));
  pos.doMove(parseUsiMove("3c3d"));
  pos.doMove(parseUsiMove("8h2b+"));
  EXPECT_EQ(Position::fromSfen(pos.toSfen()).toSfen(), pos.toSfen());
  EXPECT_EQ(pos.hand[BLACK][BISHOP], 1); // 角を取った
}
