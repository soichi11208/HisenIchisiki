#include "board.h"
#include <cassert>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace hisen {

std::string usiMove(Move m) {
  if (m == MOVE_NONE) return "resign";
  if (isDrop(m)) {
    static const char* names = " PLNSGBR";
    std::string out;
    out += names[dropType(m)];
    out += '*';
    out += usiSquare(moveTo(m));
    return out;
  }
  std::string out = usiSquare(moveFrom(m)) + usiSquare(moveTo(m));
  if (isPromo(m)) out += '+';
  return out;
}

Move parseUsiMove(const std::string& s) {
  if (s.size() < 4) return MOVE_NONE;
  auto sq = [](char f, char r) { return makeSquare(r - 'a', '9' - f); };
  if (s[1] == '*') {
    static const std::string names = " PLNSGBR";
    auto idx = names.find(s[0]);
    if (idx == std::string::npos) return MOVE_NONE;
    return makeDrop(PieceType(idx), sq(s[2], s[3]));
  }
  return makeMove(sq(s[0], s[1]), sq(s[2], s[3]), s.size() > 4 && s[4] == '+');
}

namespace {
const std::string kPieceChars = " PLNSGBK"; // sfen 用 (Kは別扱い)
char sfenChar(PieceType raw) {
  static const char* c = " plnsgbrk";
  return c[raw];
}
PieceType typeFromSfenChar(char ch) {
  switch (std::tolower(ch)) {
    case 'p': return PAWN; case 'l': return LANCE; case 'n': return KNIGHT;
    case 's': return SILVER; case 'g': return GOLD; case 'b': return BISHOP;
    case 'r': return ROOK; case 'k': return KING;
    default: throw std::runtime_error(std::string("bad sfen piece: ") + ch);
  }
}
} // namespace

Position Position::startpos() {
  return fromSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
}

Position Position::fromSfen(const std::string& sfen) {
  Position pos;
  std::istringstream ss(sfen);
  std::string boardStr, sideStr, handStr;
  ss >> boardStr >> sideStr >> handStr >> pos.ply;
  if (pos.ply <= 0) pos.ply = 1;
  pos.ply -= 1; // 内部 ply は 0 始まり

  int r = 0, f = 0;
  bool promo = false;
  for (char ch : boardStr) {
    if (ch == '/') { r++; f = 0; }
    else if (std::isdigit(ch)) { f += ch - '0'; }
    else if (ch == '+') { promo = true; }
    else {
      PieceType pt = typeFromSfenChar(ch);
      if (promo) pt = promote(pt);
      pos.board[makeSquare(r, f)] = makePiece(std::isupper(ch) ? BLACK : WHITE, pt);
      promo = false;
      f++;
    }
  }
  pos.side = (sideStr == "w") ? WHITE : BLACK;
  if (handStr != "-") {
    int count = 0;
    for (char ch : handStr) {
      if (std::isdigit(ch)) count = count * 10 + (ch - '0');
      else {
        pos.hand[std::isupper(ch) ? BLACK : WHITE][typeFromSfenChar(ch)] =
            int8_t(count == 0 ? 1 : count);
        count = 0;
      }
    }
  }
  return pos;
}

std::string Position::toSfen() const {
  std::ostringstream out;
  for (int r = 0; r < 9; r++) {
    int empty = 0;
    for (int f = 0; f < 9; f++) {
      Piece p = board[makeSquare(r, f)];
      if (!p) { empty++; continue; }
      if (empty) { out << empty; empty = 0; }
      PieceType pt = typeOf(p);
      if (isPromoted(pt)) out << '+';
      char ch = pt == KING || rawType(pt) == KING ? 'k'
              : rawType(pt) == ROOK ? 'r' : sfenChar(rawType(pt));
      out << char(colorOf(p) == BLACK ? std::toupper(ch) : ch);
    }
    if (empty) out << empty;
    if (r < 8) out << '/';
  }
  out << ' ' << (side == BLACK ? 'b' : 'w') << ' ';
  std::string hands;
  for (Color c : {BLACK, WHITE})
    for (PieceType pt : {ROOK, BISHOP, GOLD, SILVER, KNIGHT, LANCE, PAWN}) {
      int n = hand[c][pt];
      if (!n) continue;
      if (n > 1) hands += std::to_string(n);
      char ch = pt == ROOK ? 'r' : sfenChar(pt);
      hands += char(c == BLACK ? std::toupper(ch) : ch);
    }
  out << (hands.empty() ? "-" : hands) << ' ' << ply + 1;
  return out.str();
}

Square Position::kingSquare(Color c) const {
  Piece k = makePiece(c, KING);
  for (Square s = 0; s < SQ_NB; s++)
    if (board[s] == k) return s;
  return -1;
}

void Position::doMove(Move m) {
  if (isDrop(m)) {
    PieceType pt = dropType(m);
    assert(hand[side][pt] > 0 && !board[moveTo(m)]);
    hand[side][pt]--;
    board[moveTo(m)] = makePiece(side, pt);
  } else {
    Square from = moveFrom(m), to = moveTo(m);
    Piece p = board[from];
    assert(p && colorOf(p) == side);
    Piece captured = board[to];
    if (captured) hand[side][rawType(typeOf(captured))]++;
    board[from] = 0;
    board[to] = isPromo(m) ? makePiece(side, promote(typeOf(p))) : p;
  }
  side = ~side;
  ply++;
}

uint64_t Position::hashKey() const {
  uint64_t h = 1469598103934665603ull;
  auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
  for (Square s = 0; s < SQ_NB; s++) mix(uint64_t(uint8_t(board[s])) + 1);
  for (int c = 0; c < 2; c++)
    for (int t = 1; t <= 7; t++) mix(uint64_t(hand[c][t]) + 31);
  mix(side);
  return h;
}

} // namespace hisen
