#include "blocking.h"

namespace hisen {

std::vector<BlockingEdge> computeBlockingEdges(const Position& pos) {
  std::vector<BlockingEdge> out;
  for (Square s = 0; s < SQ_NB; s++) {
    Piece p = pos.board[s];
    if (!p) continue;
    PieceType pt = typeOf(p);
    if (!isSlider(pt)) continue;
    Color c = colorOf(p);
    for (auto d : slideDirs(pt, c)) {
      int r = rankOf(s) + d.dr, f = fileOf(s) + d.df;
      Square blocker = -1;
      while (onBoard(r, f)) {
        Square t = makeSquare(r, f);
        if (pos.board[t]) {
          if (blocker < 0) {
            blocker = t;
          } else {
            out.push_back({s, t, blocker, d});
            break; // 遮蔽は1枚越しまで
          }
        }
        r += d.dr; f += d.df;
      }
    }
  }
  return out;
}

} // namespace hisen
