#pragma once
#include "../engine/blocking.h"
#include "../engine/movegen.h"
#include "features.h"
#include <torch/torch.h>

namespace hisen {

// 1 局面のグラフ。テンソルは CPU 上に構築し、モデル側でデバイスへ転送。
struct Graph {
  // ノード (盤上駒 + 持ち駒 14、手番側が常に player_id=0 になるよう正規化)
  torch::Tensor node_type; // int64 [N]  駒種 ID (0..22)
  torch::Tensor node_x;    // int64 [N]
  torch::Tensor node_y;    // int64 [N]
  torch::Tensor node_scal; // float [N,4] is_promoted / player_id / is_hand / count÷7

  // エッジ (全 4 種を連結)
  torch::Tensor edge_index;    // int64 [2,E] (src,dst)
  torch::Tensor edge_type;     // int64 [E]
  torch::Tensor edge_dir;      // int64 [E]
  torch::Tensor edge_dist;     // float [E,1] チェビシェフ距離 / 8
  torch::Tensor edge_src_type; // int64 [E] 送信ノード駒種 ID

  // policy 用: 合法手 i の (srcノード, 移動先マス[手番視点], promoフラグ)
  std::vector<Move> moves;
  torch::Tensor move_src;   // int64 [M]
  torch::Tensor move_dst;   // int64 [M] 0..80
  torch::Tensor move_promo; // float [M]

  int numNodes() const { return int(node_type.size(0)); }
};

// 局面からグラフを構築。手番側視点に正規化 (後手番なら盤を 180° 回転)。
// moves を省略すると合法手を内部で生成する。
Graph buildGraph(const Position& pos);
Graph buildGraph(const Position& pos, const std::vector<Move>& legal);

} // namespace hisen
