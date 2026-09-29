#include "builder.h"
#include <cmath>

namespace hisen {

namespace {

Position normalized(const Position& pos) {
  if (pos.side == BLACK) return pos;
  Position out;
  out.side = BLACK;
  out.ply = pos.ply;
  for (Square s = 0; s < SQ_NB; s++) {
    Piece p = pos.board[s];
    out.board[80 - s] = p ? makePiece(~colorOf(p), typeOf(p)) : 0;
  }
  out.hand[BLACK] = pos.hand[WHITE];
  out.hand[WHITE] = pos.hand[BLACK];
  return out;
}

Square normSq(const Position& pos, Square s) { return pos.side == BLACK ? s : 80 - s; }

} // namespace

Graph buildGraph(const Position& pos) { return buildGraph(pos, legalMoves(pos)); }

Graph buildGraph(const Position& origPos, const std::vector<Move>& legal) {
  Position pos = normalized(origPos);
  Graph g;

  // --- ノード ---
  std::array<int, SQ_NB> nodeOfSq;
  nodeOfSq.fill(-1);
  std::vector<int64_t> ntype, nx, ny;
  std::vector<float> nscal; // 4 値/ノード
  auto addNode = [&](int type, int x, int y, float promo, float player, float isHand,
                     float count) {
    ntype.push_back(type); nx.push_back(x); ny.push_back(y);
    nscal.insert(nscal.end(), {promo, player, isHand, count});
    return int(ntype.size()) - 1;
  };

  std::vector<Square> pieceSqs;
  for (Square s = 0; s < SQ_NB; s++) {
    Piece p = pos.board[s];
    if (!p) continue;
    PieceType pt = typeOf(p);
    nodeOfSq[s] = addNode(boardPieceId(pt), fileOf(s), rankOf(s),
                          isPromoted(pt) ? 1.f : 0.f, float(colorOf(p)), 0.f, 0.f);
    pieceSqs.push_back(s);
  }
  // 持ち駒ノード 14 (先手 PAWN..ROOK、後手 PAWN..ROOK)。常に存在、枚数は特徴量。
  int handNode[2][8];
  for (int c = 0; c < 2; c++)
    for (int t = PAWN; t <= ROOK; t++)
      handNode[c][t] = addNode(handPieceId(PieceType(t)), 9 + c, t - 1, 0.f, float(c),
                               1.f, pos.hand[c][t] / 7.f);

  // --- エッジ ---
  std::vector<int64_t> esrc, edst, etype, edir, estype;
  std::vector<float> edist;
  auto addEdge = [&](int srcN, int dstN, int type, int dir, float dist) {
    esrc.push_back(srcN); edst.push_back(dstN);
    etype.push_back(type); edir.push_back(dir);
    edist.push_back(dist);
    estype.push_back(ntype[srcN]);
  };
  auto boardEdge = [&](Square a, Square b, int type) {
    int dr = rankOf(b) - rankOf(a), df = fileOf(b) - fileOf(a);
    float dist = float(std::max(std::abs(dr), std::abs(df))) / 8.f;
    addEdge(nodeOfSq[a], nodeOfSq[b], type, directionId(dr, df), dist);
  };

  // Type 0: 利き辺 (利きマスに駒がいる場合)
  for (Square s : pieceSqs)
    for (Square t : attacksFrom(pos, s))
      if (pos.board[t]) boardEdge(s, t, EDGE_KIKI);

  // Type 1: 遮蔽辺
  for (const auto& be : computeBlockingEdges(pos)) boardEdge(be.src, be.dst, EDGE_BLOCK);

  // Type 2: 近傍辺 (マンハッタン距離 ≤ 2、無向 = 双方向)
  for (size_t i = 0; i < pieceSqs.size(); i++)
    for (size_t j = i + 1; j < pieceSqs.size(); j++) {
      Square a = pieceSqs[i], b = pieceSqs[j];
      int md = std::abs(rankOf(a) - rankOf(b)) + std::abs(fileOf(a) - fileOf(b));
      if (md <= 2) { boardEdge(a, b, EDGE_NEIGHBOR); boardEdge(b, a, EDGE_NEIGHBOR); }
    }

  // Type 3: 持ち駒辺 (持ち駒ノード → 自王)
  for (int c = 0; c < 2; c++) {
    Square k = pos.kingSquare(Color(c));
    if (k < 0) continue;
    for (int t = PAWN; t <= ROOK; t++)
      addEdge(handNode[c][t], nodeOfSq[k], EDGE_HAND, 9, 1.f);
  }

  // --- policy 用 move マッピング (正規化座標) ---
  std::vector<int64_t> msrc, mdst;
  std::vector<float> mpromo;
  for (Move m : legal) {
    if (isDrop(m)) {
      msrc.push_back(handNode[BLACK][dropType(m)]);
      mpromo.push_back(0.f);
    } else {
      msrc.push_back(nodeOfSq[normSq(origPos, moveFrom(m))]);
      mpromo.push_back(isPromo(m) ? 1.f : 0.f);
    }
    mdst.push_back(normSq(origPos, moveTo(m)));
  }

  // --- テンソル化 ---
  auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  int64_t N = ntype.size(), E = esrc.size(), M = msrc.size();
  g.node_type = torch::tensor(ntype, i64);
  g.node_x = torch::tensor(nx, i64);
  g.node_y = torch::tensor(ny, i64);
  g.node_scal = torch::from_blob(nscal.data(), {N, 4}, torch::kFloat).clone();
  if (E > 0) {
    g.edge_index = torch::stack({torch::tensor(esrc, i64), torch::tensor(edst, i64)});
    g.edge_type = torch::tensor(etype, i64);
    g.edge_dir = torch::tensor(edir, i64);
    g.edge_dist = torch::from_blob(edist.data(), {E, 1}, torch::kFloat).clone();
    g.edge_src_type = torch::tensor(estype, i64);
  } else {
    g.edge_index = torch::empty({2, 0}, i64);
    g.edge_type = torch::empty({0}, i64);
    g.edge_dir = torch::empty({0}, i64);
    g.edge_dist = torch::empty({0, 1});
    g.edge_src_type = torch::empty({0}, i64);
  }
  g.moves = legal;
  g.move_src = M ? torch::tensor(msrc, i64) : torch::empty({0}, i64);
  g.move_dst = M ? torch::tensor(mdst, i64) : torch::empty({0}, i64);
  g.move_promo = M ? torch::tensor(mpromo, torch::kFloat) : torch::empty({0});
  return g;
}

} // namespace hisen
