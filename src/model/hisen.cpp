#include "hisen.h"
#include "../graph/features.h"
#include <cmath>

namespace hisen {

GraphBatch GraphBatch::collate(const std::vector<const Graph*>& graphs) {
  GraphBatch b;
  std::vector<torch::Tensor> nt, nx, ny, ns, ei, et, ed, eds, est, ms, md, mp;
  int64_t off = 0;
  for (const Graph* g : graphs) {
    b.nodeOffsets.push_back(off);
    b.sizes.push_back(g->numNodes());
    nt.push_back(g->node_type); nx.push_back(g->node_x); ny.push_back(g->node_y);
    ns.push_back(g->node_scal);
    ei.push_back(g->edge_index + off);
    et.push_back(g->edge_type); ed.push_back(g->edge_dir);
    eds.push_back(g->edge_dist); est.push_back(g->edge_src_type);
    ms.push_back(g->move_src + off);
    md.push_back(g->move_dst); mp.push_back(g->move_promo);
    b.moveCounts.push_back(g->move_src.size(0));
    off += g->numNodes();
  }
  b.node_type = torch::cat(nt); b.node_x = torch::cat(nx); b.node_y = torch::cat(ny);
  b.node_scal = torch::cat(ns);
  b.edge_index = torch::cat(ei, 1);
  b.edge_type = torch::cat(et); b.edge_dir = torch::cat(ed);
  b.edge_dist = torch::cat(eds); b.edge_src_type = torch::cat(est);
  b.move_src = torch::cat(ms); b.move_dst = torch::cat(md); b.move_promo = torch::cat(mp);
  return b;
}

GraphBatch GraphBatch::to(torch::Device dev) const {
  GraphBatch b = *this;
  for (torch::Tensor* t : {&b.node_type, &b.node_x, &b.node_y, &b.node_scal,
                           &b.edge_index, &b.edge_type, &b.edge_dir, &b.edge_dist,
                           &b.edge_src_type, &b.move_src, &b.move_dst, &b.move_promo})
    *t = t->to(dev);
  return b;
}

namespace {
// 81 マスの固定 2D sinusoidal embedding (非学習)
torch::Tensor makeSquareEmb(int dim) {
  auto emb = torch::zeros({81, dim});
  int half = dim / 2;
  for (int sq = 0; sq < 81; sq++) {
    for (int i = 0; i < half / 2; i++) {
      double freq = std::pow(10000.0, -2.0 * i / half);
      emb[sq][2 * i] = std::sin((sq % 9) * freq);
      emb[sq][2 * i + 1] = std::cos((sq % 9) * freq);
      emb[sq][half + 2 * i] = std::sin((sq / 9) * freq);
      emb[sq][half + 2 * i + 1] = std::cos((sq / 9) * freq);
    }
  }
  return emb;
}
} // namespace

TransformerLayerImpl::TransformerLayerImpl(int dim, int heads, int ffn, double dropout)
    : heads_(heads), headDim_(dim / heads), drop_(dropout) {
  qkv_ = register_module("qkv", torch::nn::Linear(dim, 3 * dim));
  proj_ = register_module("proj", torch::nn::Linear(dim, dim));
  fc1_ = register_module("fc1", torch::nn::Linear(dim, ffn));
  fc2_ = register_module("fc2", torch::nn::Linear(ffn, dim));
  norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
  norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
}

torch::Tensor TransformerLayerImpl::forward(torch::Tensor x, torch::Tensor keyPadMask) {
  int64_t B = x.size(0), S = x.size(1);
  // --- self-attention (pre-norm) ---
  auto h = norm1_(x);
  auto qkv = qkv_(h).view({B, S, 3, heads_, headDim_}).permute({2, 0, 3, 1, 4});
  auto q = qkv[0], k = qkv[1], v = qkv[2]; // [B,heads,S,headDim]
  // padding キーを無効化する bool マスク [B,1,1,S] (true = attend 可)
  std::optional<torch::Tensor> attnMask;
  if (keyPadMask.defined())
    attnMask = keyPadMask.logical_not().view({B, 1, 1, S});
  double dp = is_training() ? drop_ : 0.0;
  auto attn = at::scaled_dot_product_attention(q, k, v, attnMask, dp);
  attn = attn.permute({0, 2, 1, 3}).reshape({B, S, heads_ * headDim_});
  x = x + torch::dropout(proj_(attn), drop_, is_training());
  // --- FFN (pre-norm) ---
  auto f = fc2_(torch::dropout(torch::gelu(fc1_(norm2_(x))), drop_, is_training()));
  return x + torch::dropout(f, drop_, is_training());
}

HisenNetImpl::HisenNetImpl(int dim, int gatLayers, int tfLayers, int heads, int ffn)
    : dim_(dim) { // 既定値はヘッダ参照 (30M 構成)
  pieceEmb_ = register_module("piece_emb", torch::nn::Embedding(kPieceTypeVocab, 32));
  xEmb_ = register_module("x_emb", torch::nn::Embedding(kXVocab, 16));
  yEmb_ = register_module("y_emb", torch::nn::Embedding(kYVocab, 16));
  scalProj_ = register_module("scal_proj", torch::nn::Linear(4, 8));
  nodeProj_ = register_module("node_proj", torch::nn::Linear(72, dim));

  eTypeEmb_ = register_module("etype_emb", torch::nn::Embedding(EDGE_TYPE_NB, 16));
  eDirEmb_ = register_module("edir_emb", torch::nn::Embedding(kDirVocab, 16));
  eSrcEmb_ = register_module("esrc_emb", torch::nn::Embedding(kPieceTypeVocab, 16));
  eDistProj_ = register_module("edist_proj", torch::nn::Linear(1, 4));
  edgeProj_ = register_module("edge_proj", torch::nn::Linear(52, 64));

  for (int i = 0; i < gatLayers; i++)
    gat_.push_back(register_module("gat" + std::to_string(i), GATv2Layer(dim, heads, 64)));

  for (int i = 0; i < tfLayers; i++)
    tf_.push_back(register_module("tf" + std::to_string(i),
                                  TransformerLayer(dim, heads, ffn, 0.1)));

  squareEmb_ = register_buffer("square_emb", makeSquareEmb(dim));
  promoHead_ = register_module("promo_head", torch::nn::Linear(dim, 1));
  value1_ = register_module("value1", torch::nn::Linear(dim, dim));
  value2_ = register_module("value2", torch::nn::Linear(dim, 1));
}

std::pair<torch::Tensor, torch::Tensor> HisenNetImpl::forward(const GraphBatch& b) {
  // --- Embedding ---
  auto nodeFeat = torch::cat({pieceEmb_(b.node_type),
                              xEmb_(b.node_x), yEmb_(b.node_y),
                              scalProj_(b.node_scal)}, 1); // [N,72]
  auto h = nodeProj_(nodeFeat); // [N,dim]
  auto eattr = edgeProj_(torch::cat({eTypeEmb_(b.edge_type), eDirEmb_(b.edge_dir),
                                     eDistProj_(b.edge_dist),
                                     eSrcEmb_(b.edge_src_type)}, 1)); // [E,64]

  // --- GATv2 ---
  for (auto& layer : gat_) h = layer->forward(h, b.edge_index, eattr);

  // --- Transformer (padding でバッチ化、[S,B,dim]) ---
  int64_t B = b.sizes.size();
  int64_t maxN = *std::max_element(b.sizes.begin(), b.sizes.end());
  auto padded = torch::zeros({B, maxN, dim_}, h.options());
  auto mask = torch::ones({B, maxN}, torch::TensorOptions().dtype(torch::kBool).device(h.device()));
  for (int64_t i = 0; i < B; i++) {
    padded.index_put_({i, torch::indexing::Slice(0, b.sizes[i])},
                      h.narrow(0, b.nodeOffsets[i], b.sizes[i]));
    mask.index_put_({i, torch::indexing::Slice(0, b.sizes[i])}, false);
  }
  auto out = padded; // [B,maxN,dim]
  for (auto& layer : tf_) out = layer->forward(out, mask);

  // 連結形式 [N,dim] に戻す
  std::vector<torch::Tensor> parts;
  for (int64_t i = 0; i < B; i++)
    parts.push_back(out[i].narrow(0, 0, b.sizes[i]));
  h = torch::cat(parts); // [N,dim]

  // --- Policy: dot(h_src, square_emb[dst]) + promo * promoHead(h_src) ---
  auto hSrc = h.index_select(0, b.move_src);                  // [M,dim]
  auto sqE = squareEmb_.index_select(0, b.move_dst);          // [M,dim]
  auto logits = (hSrc * sqE).sum(1) / std::sqrt(double(dim_)) +
                b.move_promo * promoHead_(hSrc).squeeze(1);   // [M]

  // --- Value: グラフごとの mean pool → MLP → tanh ---
  std::vector<torch::Tensor> pooled;
  for (int64_t i = 0; i < B; i++)
    pooled.push_back(h.narrow(0, b.nodeOffsets[i], b.sizes[i]).mean(0));
  auto v = torch::tanh(value2_(torch::relu(value1_(torch::stack(pooled))))).squeeze(1);
  return {logits, v};
}

std::vector<torch::Tensor> splitPolicyLogSoftmax(torch::Tensor logits,
                                                 const std::vector<int64_t>& moveCounts) {
  std::vector<torch::Tensor> out;
  int64_t off = 0;
  for (int64_t m : moveCounts) {
    out.push_back(torch::log_softmax(logits.narrow(0, off, m), 0));
    off += m;
  }
  return out;
}

} // namespace hisen
