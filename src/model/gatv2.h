#pragma once
#include <torch/torch.h>

namespace hisen {

// GATv2 層 (エッジ特徴量つき動的アテンション、自前実装)。
// score_e = a_h^T LeakyReLU(Ws h_src + Wt h_dst + We e)、softmax は dst ごと。
struct GATv2LayerImpl : torch::nn::Module {
  GATv2LayerImpl(int dim, int heads, int edgeDim);

  // x: [N,dim], edge_index: int64 [2,E], eattr: [E,edgeDim] → [N,dim]
  // Residual + LayerNorm 込み。
  torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index, torch::Tensor eattr);

  int heads_, headDim_;
  torch::nn::Linear ws_{nullptr}, wt_{nullptr}, we_{nullptr};
  torch::nn::Linear wv_{nullptr}, wev_{nullptr}, wo_{nullptr};
  torch::Tensor att_; // [H, headDim]
  torch::nn::LayerNorm norm_{nullptr};
};
TORCH_MODULE(GATv2Layer);

} // namespace hisen
