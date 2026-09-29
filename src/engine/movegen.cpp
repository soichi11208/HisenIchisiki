#include "movegen.h"
#include "kiki.h"
#include <algorithm>

namespace hisen {

namespace {

bool inPromoZone(Color c, Square sq) {
  int r = rankOf(sq);
  return c == BLACK ? r <= 2 : r >= 6;
}

// 行き所のない駒
bool isDeadSquare(Color c, PieceType pt, Square sq) {
  int r = rankOf(sq);
  int last = (c == BLACK) ? 0 : 8;
  if ((pt == PAWN || pt == LANCE) && r == last) return true;
  if (pt == KNIGHT) {
    int last2 = (c == BLACK) ? 1 : 7;
    return c == BLACK ? r <= last2 : r >= last2;
  }
  return false;
}

bool leavesKingInCheck(const Position& pos, Move m) {
  Position next = pos;
  next.doMove(m);
  return inCheck(next, pos.side);
}

// 二歩
bool hasPawnOnFile(const Position& pos, Color c, int file) {
  for (int r = 0; r < 9; r++)
    if (pos.board[makeSquare(r, file)] == makePiece(c, PAWN)) return true;
  return false;
}

std::vector<Move> pseudoLegalNoDropRules(const Position& pos);

// 打ち歩詰め
bool isPawnDropMate(const Position& pos, Move m) {
  Position next = pos;
  next.doMove(m);
  if (!inCheck(next, next.side)) return false;
  for (Move r : pseudoLegalNoDropRules(next)) {
    if (!leavesKingInCheck(next, r)) return false;
  }
  return true;
}

std::vector<Move> pseudoLegalNoDropRules(const Position& pos) {
  std::vector<Move> moves;
  moves.reserve(128);
  Color us = pos.side;

  // 盤上の駒の移動
  for (Square from = 0; from < SQ_NB; from++) {
    Piece p = pos.board[from];
    if (!p || colorOf(p) != us) continue;
    PieceType pt = typeOf(p);
    for (Square to : attacksFrom(pos, from)) {
      Piece cap = pos.board[to];
      if (cap && colorOf(cap) == us) continue;
      bool canPromo = canPromoteType(pt) &&
                      (inPromoZone(us, from) || inPromoZone(us, to));
      bool mustPromo = isDeadSquare(us, pt, to);
      if (!mustPromo) moves.push_back(makeMove(from, to, false));
      if (canPromo) moves.push_back(makeMove(from, to, true));
    }
  }

  // 持ち駒打ち
  for (int t = PAWN; t <= ROOK; t++) {
    if (!pos.hand[us][t]) continue;
    PieceType pt = PieceType(t);
    for (Square to = 0; to < SQ_NB; to++) {
      if (pos.board[to]) continue;
      if (isDeadSquare(us, pt, to)) continue;
      if (pt == PAWN && hasPawnOnFile(pos, us, fileOf(to))) continue;
      moves.push_back(makeDrop(pt, to));
    }
  }
  return moves;
}

} // namespace

std::vector<Move> legalMoves(const Position& pos) {
  std::vector<Move> out;
  for (Move m : pseudoLegalNoDropRules(pos)) {
    if (leavesKingInCheck(pos, m)) continue;
    if (isDrop(m) && dropType(m) == PAWN && isPawnDropMate(pos, m)) continue;
    out.push_back(m);
  }
  return out;
}

bool isLegal(const Position& pos, Move m) {
  auto ms = legalMoves(pos);
  return std::find(ms.begin(), ms.end(), m) != ms.end();
}

} // namespace hisen
