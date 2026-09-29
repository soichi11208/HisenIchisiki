#include "mate.h"
#include "kiki.h"
#include "movegen.h"
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <vector>

namespace hisen {

namespace {

// 置換表: key = hash ^ (depth*定数)。value: 1=詰む, 0=詰まない。
// thread_local で並列 MCTS ワーカーごとに独立 (ロック不要)。
// 他スレッドから clear できないため、増え過ぎたら自分で捨てる (上限でメモリを抑える)。
thread_local std::unordered_map<uint64_t, int8_t> g_mateTT;
constexpr size_t kMateTTMaxEntries = 1u << 18; // 262144 エントリ/スレッド

inline void mateTTStore(uint64_t key, int8_t v) {
  if (g_mateTT.size() >= kMateTTMaxEntries) g_mateTT.clear();
  g_mateTT[key] = v;
}

uint64_t mateKey(uint64_t h, int depth) {
  return h ^ (uint64_t(depth) * 0x9E3779B97F4A7C15ull);
}

// 詰み探索の打ち切り判定。32 ノードごとに stopFlag / 締切 / ノード上限を見る
// (1 ノードの計算が重いので、この間隔ならコストは無視でき、24 スレッド分の
//  粒度ボケによる超過も小さく抑えられる)。制限に達したら true を返し、
// 呼び出し側は「詰み無し」に倒して置換表へは書かない。
struct MateCtx {
  const MateAbort* abort = nullptr;
  int64_t nodes = 0;
  int64_t nextCheck = 32;
  bool aborted = false;

  bool checkAbort() {
    if (aborted) return true;
    if (++nodes < nextCheck) return false;
    nextCheck += 32;
    if (!abort) return false;
    if (abort->stopFlag && abort->stopFlag->load()) {
      aborted = true;
    } else if (abort->nodeBudget > 0 && nodes >= abort->nodeBudget) {
      aborted = true;
    } else if (abort->deadlineTicks > 0 &&
               std::chrono::steady_clock::now().time_since_epoch().count() >=
                   abort->deadlineTicks) {
      aborted = true;
    }
    return aborted;
  }
};

// 駒のざっくり価値 (オーダリング用)
int pieceVal(PieceType pt) {
  switch (rawType(pt)) {
    case PAWN: return 1;
    case LANCE: case KNIGHT: return 3;
    case SILVER: return 5;
    case GOLD: return 6;
    case BISHOP: return 8;
    case ROOK: return 10;
    default: return isPromoted(pt) ? 6 : 0;
  }
}

// 攻め: 王手を優先。取る駒が大きい・成る手を先に。
void orderAttackMoves(const Position& pos, std::vector<Move>& moves) {
  std::vector<std::pair<int, Move>> scored;
  scored.reserve(moves.size());
  for (Move m : moves) {
    Position next = pos;
    next.doMove(m);
    if (!inCheck(next, next.side)) continue; // 王手のみ
    int score = 1000;
    if (!isDrop(m)) {
      Piece cap = pos.board[moveTo(m)];
      if (cap) score += 10 * pieceVal(typeOf(cap));
    }
    if (isPromo(m)) score += 5;
    // 玉に近い打ち/寄りを少し優先
    Square ksq = next.kingSquare(next.side);
    int dist = std::abs(rankOf(moveTo(m)) - rankOf(ksq)) +
               std::abs(fileOf(moveTo(m)) - fileOf(ksq));
    score += std::max(0, 8 - dist);
    scored.push_back({score, m});
  }
  std::sort(scored.begin(), scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  moves.clear();
  for (auto& [s, m] : scored) moves.push_back(m);
}

// 受け: 王手を解消する取り・安い駒の移動を先に (早期の「逃れ」検出)
void orderDefenseMoves(const Position& pos, std::vector<Move>& moves) {
  std::vector<std::pair<int, Move>> scored;
  scored.reserve(moves.size());
  for (Move m : moves) {
    int score = 0;
    if (!isDrop(m)) {
      Piece cap = pos.board[moveTo(m)];
      if (cap) score += 20 * pieceVal(typeOf(cap));
      Piece me = pos.board[moveFrom(m)];
      if (me) score -= pieceVal(typeOf(me)); // 安い駒で受ける
    } else {
      score += 2;
    }
    if (isPromo(m)) score += 3;
    scored.push_back({score, m});
  }
  std::sort(scored.begin(), scored.end(),
            [](auto& a, auto& b) { return a.first > b.first; });
  moves.clear();
  for (auto& [s, m] : scored) moves.push_back(m);
}

bool isMated(const Position& pos, int depth, MateCtx& ctx);

bool canForceMateImpl(const Position& pos, int depth, Move* firstMove, MateCtx& ctx) {
  if (depth <= 0) return false;
  if (ctx.checkAbort()) return false; // 打ち切り: 詰み無しに倒す

  uint64_t key = mateKey(pos.hashKey(), depth);
  if (!firstMove) {
    auto it = g_mateTT.find(key);
    if (it != g_mateTT.end()) return it->second != 0;
  }

  auto moves = legalMoves(pos);
  orderAttackMoves(pos, moves); // 王手のみ残る

  for (Move m : moves) {
    Position next = pos;
    next.doMove(m);
    // orderAttackMoves で王手のみに絞済み
    if (isMated(next, depth - 1, ctx)) {
      if (firstMove) *firstMove = m;
      if (!firstMove) mateTTStore(key, 1);
      return true;
    }
    if (ctx.aborted) return false; // 途中で打ち切り: 置換表へは書かない
  }
  if (!firstMove && !ctx.aborted) mateTTStore(key, 0);
  return false;
}

bool isMated(const Position& pos, int depth, MateCtx& ctx) {
  auto moves = legalMoves(pos);
  if (moves.empty()) return inCheck(pos, pos.side);
  if (depth <= 0) return false;
  if (ctx.checkAbort()) return false; // 打ち切り: 「詰まされていない」に倒す

  uint64_t key = mateKey(pos.hashKey(), depth) ^ 0xA5A5A5A5A5A5A5A5ull;
  auto it = g_mateTT.find(key);
  if (it != g_mateTT.end()) return it->second != 0;

  orderDefenseMoves(pos, moves);
  for (Move m : moves) {
    Position next = pos;
    next.doMove(m);
    if (!canForceMateImpl(next, depth, nullptr, ctx)) {
      if (!ctx.aborted) mateTTStore(key, 0);
      return false;
    }
    if (ctx.aborted) return false;
  }
  if (!ctx.aborted) mateTTStore(key, 1);
  return true;
}

} // namespace

void clearMateTable() { g_mateTT.clear(); }

bool isCheckmated(const Position& pos) {
  return legalMoves(pos).empty() && inCheck(pos, pos.side);
}

bool canForceMate(const Position& pos, int depth, Move* firstMove,
                  const MateAbort* abort) {
  if (depth <= 0) return false;
  // firstMove 要求時はルート用に TT ヒットで手を返せないのでフル探索
  MateCtx ctx;
  ctx.abort = abort;
  return canForceMateImpl(pos, depth, firstMove, ctx);
}

} // namespace hisen
