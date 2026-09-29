#pragma once
#include "types.h"
#include <array>

namespace hisen {

struct Position {
  std::array<Piece, SQ_NB> board{};
  // hand[color][rawType] (PAWN..ROOK = 1..7)
  std::array<std::array<int8_t, 8>, 2> hand{};
  Color side = BLACK;
  int ply = 0;

  static Position startpos();
  static Position fromSfen(const std::string& sfen);
  std::string toSfen() const;

  Square kingSquare(Color c) const;
  void doMove(Move m); // 合法性は呼び出し側が保証

  bool operator==(const Position& o) const {
    return board == o.board && hand == o.hand && side == o.side;
  }
  uint64_t hashKey() const;
};

} // namespace hisen
