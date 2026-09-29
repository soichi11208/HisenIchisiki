#pragma once
#include "board.h"
#include <atomic>
#include <cstdint>

namespace hisen {

// 詰み探索の打ち切り条件。nullptr または 0 の項目は無効 (= 無制限)。
// 停止フラグ/締切を渡さない従来呼び出しはそのまま動く (abort = nullptr)。
struct MateAbort {
  const std::atomic<bool>* stopFlag = nullptr; // 監視する停止フラグ
  int64_t deadlineTicks = 0; // steady_clock の time_since_epoch().count()。0 で無効
  int64_t nodeBudget = 0;    // この呼び出しで探索する最大ノード数。0 で無効
};

// 手番側が depth 手以内に相手を詰ませられるか (depth は攻め方の手数)。
// 攻め方は王手のみ。置換表 + 手のオーダリング付き。
// 浅い depth で MCTS のリーフ/ルートに差し、戦術的な詰みの見落としを防ぐ。
// abort が尽きた場合は「詰み無し」に倒して打ち切る (誤った詰みを報告しない)。
bool canForceMate(const Position& pos, int depth, Move* firstMove = nullptr,
                  const MateAbort* abort = nullptr);

// 探索開始時に置換表を空にする (局面跨ぎの誤ヒット防止)。
void clearMateTable();

// 手番側が今詰んでいるか (合法手なし + 王手)。
bool isCheckmated(const Position& pos);

} // namespace hisen
