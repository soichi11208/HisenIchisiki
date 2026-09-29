#pragma once
#include "board.h"
#include <vector>

namespace hisen {

struct Step { int dr, df; };

const std::vector<Step>& stepMoves(PieceType pt, Color c);
const std::vector<Step>& slideDirs(PieceType pt, Color c);

std::vector<Square> attacksFrom(const Position& pos, Square sq);

bool isAttacked(const Position& pos, Square sq, Color by);

bool inCheck(const Position& pos, Color c);

} // namespace hisen
