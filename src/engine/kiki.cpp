#include "kiki.h"

namespace hisen {

namespace {
const std::vector<Step> kGold = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,0}};
const std::vector<Step> kKing = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
const std::vector<Step> kOrth = {{-1,0},{1,0},{0,-1},{0,1}};
const std::vector<Step> kDiag = {{-1,-1},{-1,1},{1,-1},{1,1}};
const std::vector<Step> kEmpty = {};

std::vector<Step> flip(const std::vector<Step>& v) {
  std::vector<Step> out;
  for (auto s : v) out.push_back({-s.dr, s.df});
  return out;
}
} // namespace

const std::vector<Step>& stepMoves(PieceType pt, Color c) {
  static const std::vector<Step> pawnB = {{-1,0}}, pawnW = flip(pawnB);
  static const std::vector<Step> knightB = {{-2,-1},{-2,1}}, knightW = flip(knightB);
  static const std::vector<Step> silverB = {{-1,-1},{-1,0},{-1,1},{1,-1},{1,1}},
                                 silverW = flip(silverB);
  static const std::vector<Step> goldW = flip(kGold);
  switch (pt) {
    case PAWN: return c == BLACK ? pawnB : pawnW;
    case KNIGHT: return c == BLACK ? knightB : knightW;
    case SILVER: return c == BLACK ? silverB : silverW;
    case GOLD: case PRO_PAWN: case PRO_LANCE: case PRO_KNIGHT: case PRO_SILVER:
      return c == BLACK ? kGold : goldW;
    case KING: return kKing;
    case HORSE: return kOrth;   // 馬 = 角スライド + 縦横1
    case DRAGON: return kDiag;  // 龍 = 飛スライド + 斜め1
    default: return kEmpty;
  }
}

const std::vector<Step>& slideDirs(PieceType pt, Color c) {
  static const std::vector<Step> lanceB = {{-1,0}}, lanceW = flip(lanceB);
  switch (pt) {
    case LANCE: return c == BLACK ? lanceB : lanceW;
    case BISHOP: case HORSE: return kDiag;
    case ROOK: case DRAGON: return kOrth;
    default: return kEmpty;
  }
}

std::vector<Square> attacksFrom(const Position& pos, Square sq) {
  std::vector<Square> out;
  Piece p = pos.board[sq];
  if (!p) return out;
  PieceType pt = typeOf(p);
  Color c = colorOf(p);
  int r0 = rankOf(sq), f0 = fileOf(sq);
  for (auto s : stepMoves(pt, c)) {
    int r = r0 + s.dr, f = f0 + s.df;
    if (onBoard(r, f)) out.push_back(makeSquare(r, f));
  }
  for (auto d : slideDirs(pt, c)) {
    int r = r0 + d.dr, f = f0 + d.df;
    while (onBoard(r, f)) {
      out.push_back(makeSquare(r, f));
      if (pos.board[makeSquare(r, f)]) break;
      r += d.dr; f += d.df;
    }
  }
  return out;
}

bool isAttacked(const Position& pos, Square sq, Color by) {
  for (Square s = 0; s < SQ_NB; s++) {
    Piece p = pos.board[s];
    if (!p || colorOf(p) != by) continue;
    for (Square t : attacksFrom(pos, s))
      if (t == sq) return true;
  }
  return false;
}

bool inCheck(const Position& pos, Color c) {
  Square k = pos.kingSquare(c);
  return k >= 0 && isAttacked(pos, k, ~c);
}

} // namespace hisen
