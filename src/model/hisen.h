#pragma once
#include "../graph/builder.h"
#include "gatv2.h"

namespace hisen {

// 複数グラフを 1 バッチに連結したもの (ノード index はオフセット済み)
struct GraphBatch {
  torch::Tensor node_type, node_x, node_y, node_scal;
  torch::Tensor edge_index, edge_type, edge_dir, edge_dist, edge_src_type;
  std::vector<int64_t> sizes;        // 各グラフのノード数
  std::vector<int64_t> nodeOffsets;  // 各グラフの先頭ノード index
  // policy 用 (連結、ノードオフセット適用済み)
  torch::Tensor move_src, move_dst, move_promo;
  std::vector<int64_t> moveCounts;

  static GraphBatch collate(const std::vector<const Graph*>& graphs);
  GraphBatch to(torch::Device dev) const;
};

// Pre-norm Transformer エンコーダ層 (自前実装)。
// libtorch の nn::TransformerEncoder は bf16 autocast + padding マスクで
// masked_fill 未実装エラーになるため、SDPA (Flash Attention 対応) で実装する。
struct TransformerLayerImpl : torch::nn::Module {
  TransformerLayerImpl(int dim, int heads, int ffn, double dropout);
  // x: [B,S,dim], keyPadMask: bool [B,S] (true = padding)
  torch::Tensor forward(torch::Tensor x, torch::Tensor keyPadMask);

  int heads_, headDim_;
  double drop_;
  torch::nn::Linear qkv_{nullptr}, proj_{nullptr}, fc1_{nullptr}, fc2_{nullptr};
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
};
TORCH_MODULE(TransformerLayer);

struct HisenNetImpl : torch::nn::Module {
  // 既定 = 30M 構成 (dim512 / GATv2 4層 / Transformer 8層 / 8-head / FFN2048)。
  // 全バイナリがこの既定でモデルを構築するため、変更時はチェックポイント互換性が切れる。
  HisenNetImpl(int dim = 512, int gatLayers = 4, int tfLayers = 8, int heads = 8,
                int ffn = 2048);

  // 戻り値: {policy_logits [Mtotal] (グラフごとに softmax すべき), value [B]}
  std::pair<torch::Tensor, torch::Tensor> forward(const GraphBatch& b);

  int dim_;
  torch::nn::Embedding pieceEmb_{nullptr}, xEmb_{nullptr}, yEmb_{nullptr};
  torch::nn::Linear scalProj_{nullptr}, nodeProj_{nullptr};
  torch::nn::Embedding eTypeEmb_{nullptr}, eDirEmb_{nullptr}, eSrcEmb_{nullptr};
  torch::nn::Linear eDistProj_{nullptr}, edgeProj_{nullptr};
  std::vector<GATv2Layer> gat_;
  std::vector<TransformerLayer> tf_;
  torch::Tensor squareEmb_; // [81,dim] 固定 (非学習)
  torch::nn::Linear promoHead_{nullptr};
  torch::nn::Linear value1_{nullptr}, value2_{nullptr};
};
TORCH_MODULE(HisenNet);

// グラフごとの log_softmax 済み policy を返すユーティリティ
// (logits をグラフ単位に分割して log_softmax)
std::vector<torch::Tensor> splitPolicyLogSoftmax(torch::Tensor logits,
                                                 const std::vector<int64_t>& moveCounts);

} // namespace hisen
