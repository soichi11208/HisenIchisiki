#include "gatv2.h"

namespace hisen {

GATv2LayerImpl::GATv2LayerImpl(int dim, int heads, int edgeDim)
    : heads_(heads), headDim_(dim / heads) {
  ws_ = register_module("ws", torch::nn::Linear(dim, dim));
  wt_ = register_module("wt", torch::nn::Linear(dim, dim));
  we_ = register_module("we", torch::nn::Linear(edgeDim, dim));
  wv_ = register_module("wv", torch::nn::Linear(dim, dim));
  wev_ = register_module("wev", torch::nn::Linear(edgeDim, dim));
  wo_ = register_module("wo", torch::nn::Linear(dim, dim));
  att_ = register_parameter("att", torch::randn({heads_, headDim_}) * 0.1);
  norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
}

torch::Tensor GATv2LayerImpl::forward(torch::Tensor x, torch::Tensor edge_index,
                                      torch::Tensor eattr) {
  int64_t N = x.size(0), E = edge_index.size(1);
  if (E == 0) return norm_(x);
  auto src = edge_index[0], dst = edge_index[1];

  auto view = [&](torch::Tensor t) { return t.view({-1, heads_, headDim_}); };
  auto m = view(ws_(x)).index_select(0, src) + view(wt_(x)).index_select(0, dst) +
           view(we_(eattr)); // [E,H,D]
  auto score = (torch::leaky_relu(m, 0.2) * att_.unsqueeze(0)).sum(-1); // [E,H]

  auto dstH = dst.unsqueeze(1).expand({E, heads_});
  auto maxs = torch::full({N, heads_}, -1e30, score.options())
                  .scatter_reduce_(0, dstH, score, "amax", /*include_self=*/true);
  auto ex = torch::exp(score - maxs.index_select(0, dst));
  auto denom = torch::zeros({N, heads_}, ex.options()).index_add_(0, dst, ex);
  auto alpha = ex / (denom.index_select(0, dst) + 1e-16); // [E,H]

  auto msg = view(wv_(x)).index_select(0, src) + view(wev_(eattr)); // [E,H,D]
  auto weighted = (msg * alpha.unsqueeze(-1)).view({E, heads_ * headDim_});
  auto out =
      torch::zeros({N, heads_ * headDim_}, weighted.options()).index_add_(0, dst, weighted);
  return norm_(x + wo_(out));
}

} // namespace hisen
