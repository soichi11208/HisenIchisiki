#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace hisen {

enum Color : int8_t { BLACK = 0, WHITE = 1, COLOR_NONE = 2 };
inline Color operator~(Color c) { return Color(c ^ 1); }

// 駒種。9以上が成駒 (promote = +8)。13(成金)は欠番。
enum PieceType : int8_t {
  NO_PIECE_TYPE = 0,
  PAWN = 1, LANCE = 2, KNIGHT = 3, SILVER = 4, GOLD = 5, BISHOP = 6, ROOK = 7, KING = 8,
  PRO_PAWN = 9, PRO_LANCE = 10, PRO_KNIGHT = 11, PRO_SILVER = 12,
  HORSE = 14, DRAGON = 15,
  PIECE_TYPE_NB = 16
};

inline bool isPromoted(PieceType pt) { return pt >= PRO_PAWN; }
inline PieceType promote(PieceType pt) { return PieceType(pt + 8); }
inline PieceType rawType(PieceType pt) { return pt >= PRO_PAWN ? PieceType(pt - 8) : pt; }
inline bool canPromoteType(PieceType pt) {
  return pt == PAWN || pt == LANCE || pt == KNIGHT || pt == SILVER || pt == BISHOP || pt == ROOK;
}
inline bool isSlider(PieceType pt) {
  return pt == LANCE || pt == BISHOP || pt == ROOK || pt == HORSE || pt == DRAGON;
}

// Piece = type | (color << 4)。0 = 空マス。
using Piece = int8_t;
inline Piece makePiece(Color c, PieceType pt) { return Piece(pt | (c << 4)); }
inline PieceType typeOf(Piece p) { return PieceType(p & 15); }
inline Color colorOf(Piece p) { return Color(p >> 4); }

// マス: sq = rank*9 + file。rank 0 = 一段目 (SFEN 'a')、file 0 = 9筋 (盤面左端)。
// 先手 (BLACK) は rank が減る方向に進む。
using Square = int;
constexpr Square SQ_NB = 81;
inline int rankOf(Square s) { return s / 9; }
inline int fileOf(Square s) { return s % 9; }
inline bool onBoard(int r, int f) { return r >= 0 && r < 9 && f >= 0 && f < 9; }
inline Square makeSquare(int r, int f) { return r * 9 + f; }
// USI 表記: file 0 → '9'、rank 0 → 'a'
inline std::string usiSquare(Square s) {
  std::string out;
  out += char('9' - fileOf(s));
  out += char('a' + rankOf(s));
  return out;
}

// Move エンコード: bits 0-6 to、bits 7-13 from (打ちは 81+rawType)、bit 14 promo。
using Move = uint32_t;
constexpr Move MOVE_NONE = 0;
inline Move makeMove(Square from, Square to, bool promo) {
  return Move(to | (from << 7) | (uint32_t(promo) << 14));
}
inline Move makeDrop(PieceType pt, Square to) { return Move(to | ((81 + pt) << 7)); }
inline Square moveTo(Move m) { return m & 0x7f; }
inline int moveFromRaw(Move m) { return (m >> 7) & 0x7f; }
inline bool isDrop(Move m) { return moveFromRaw(m) >= 81; }
inline Square moveFrom(Move m) { return moveFromRaw(m); }
inline PieceType dropType(Move m) { return PieceType(moveFromRaw(m) - 81); }
inline bool isPromo(Move m) { return (m >> 14) & 1; }

std::string usiMove(Move m);
Move parseUsiMove(const std::string& s);

} // namespace hisen
