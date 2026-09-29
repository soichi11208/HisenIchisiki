#pragma once
#include "kiki.h"
#include <vector>

namespace hisen {

struct BlockingEdge {
  Square src; // 長距離駒
  Square dst; // 遮蔽駒の先にいる駒
  Square blocker;
  Step dir; // src 視点の方向
};

std::vector<BlockingEdge> computeBlockingEdges(const Position& pos);

} // namespace hisen
