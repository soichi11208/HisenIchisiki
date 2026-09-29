#pragma once
#include "../engine/types.h"

namespace hisen {

constexpr int kPieceTypeVocab = 23;

inline int boardPieceId(PieceType pt) {
  if (pt <= KING) return pt - 1;
  if (pt <= PRO_SILVER) return pt - 1; // 9..12 → 8..11
  return pt == HORSE ? 12 : 13;
}
inline int handPieceId(PieceType raw) { return 14 + (raw - 1); } // PAWN..ROOK

// エッジ種別
enum EdgeType { EDGE_KIKI = 0, EDGE_BLOCK = 1, EDGE_NEIGHBOR = 2, EDGE_HAND = 3, EDGE_TYPE_NB = 4 };

// 方向 ID: 8方位 0..7 (N,NE,E,SE,S,SW,W,NW)、桂・非整列 = 8、持ち駒辺 = 9
constexpr int kDirVocab = 10;
inline int directionId(int dr, int df) {
  if (dr == 0 && df == 0) return 8;
  int sr = (dr > 0) - (dr < 0), sf = (df > 0) - (df < 0);
  bool aligned = dr == 0 || df == 0 || (dr == sr * std::abs(df) * (df == 0 ? 1 : 1) && std::abs(dr) == std::abs(df));
  if (!(dr == 0 || df == 0 || std::abs(dr) == std::abs(df))) return 8;
  static const int table[3][3] = {{7, 0, 1}, {6, 8, 2}, {5, 4, 3}}; // [sr+1][sf+1]
  (void)aligned;
  return table[sr + 1][sf + 1];
}

// 持ち駒ノードの仮想座標: x=9, y = 駒種 index (先手) / 駒種+7 (後手は x=10)
constexpr int kXVocab = 11; // file 0..8 + 仮想 9,10
constexpr int kYVocab = 14; // rank 0..8 + 持ち駒 index 用余白

} // namespace hisen
