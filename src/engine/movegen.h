#pragma once
#include "board.h"
#include <vector>

namespace hisen {

// 完全な合法手 (自玉放置・二歩・行き所のない駒・打ち歩詰めを除外)。
// 千日手は探索側で扱う。
std::vector<Move> legalMoves(const Position& pos);

bool isLegal(const Position& pos, Move m);

} // namespace hisen
