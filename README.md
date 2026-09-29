# 緋閃壱式(Hisen Ichishiki)

GNN(GATv2)+Transformer+MCTSによる将棋モデルです。

たったの30Mパラメータ(LLMに感覚を麻痺させられているだけである)

ここで配布するのがエンジンであり、ここ(https://huggingface.co/soichi1208/HisenIchisiki )にて重みも公開しています。

重みにはRLを行っていないBaseと行ったRLがありますが、強さはほぼ互角です。

- 全コードC++17/libtorch
- GPU推論(CUDA)+マルチスレッドMCTSでbf16です。つまりGPUとCPU両方ともハードに使います。VRAM帯域よりチップが重要だったりする。
- モデル重みは同梱していないので、USIのモデルパスで渡してください。

## 要件

- CMake3.18以上
- GCC/G++14(CUDA12.8搭載のlibtorchでは、GCC16系の最新コンパイラをnvccが拒否するため14指定が必須)
- pipのtorch(2.9+/cu128系)とNVIDIA GPU(無い場合はCPU動作するが遅い)
- どのOSでも動くはずですが、動作検証はDebain SidのLinux7.12でのみ行っていますし、Linuxを推奨しています。

## ビルド

コンパイラはPATH上の名前を指定。gcc-14がPATHに無い環境はフルパスを指定してください。

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-14 \
      -DCMAKE_CXX_COMPILER=g++-14 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
      -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
make -j$(nproc)
ctest --output-on-failure   # 全テスト (オフにする場合: cmake -DHISEN_BUILD_TESTS=OFF ..)
```

生成バイナリ: build/hisen-usi

## 重みの準備

モデル重み(.pt)は同梱されていません。

相対パスの解決順:指定された通り(実行cwd基準)→実行ファイル隣→実行ファイル隣のweights/。
見つからない場合はランダム重みで動かさずinfo string ERROR model NOT FOUND ... を出しgoにはbestmove resignで応答します(GUIのログにERROR行が残る)。

## 対局(USI)

ShogiHome等にエンジン登録:<このディレクトリ>/build/hisen-usi

みんなもShogiHomeで痛将棋盤やろう。

### エンジンオプション (USI)

| オプション         | 推奨                      | 説明                           |
| ------------- | ----------------------- | ---------------------------- |
| ModelPath     | weights/hisen_latest.pt | 重み。                          |
| Nodes         | 4000                    | 時間制御が無いときの固定探索ノード数           |
| Threads       | CPUの論理コア数               | 並列探索スレッド数                    |
| BatchSize     | 64                      | GPU推論バッチサイズ                  |
| MateDepth     | 5                       | ルート/リーフの詰み探索手数(0で無効)         |
| TacticalDepth | 0                       | リーフ戦術qsearch深さ(既定0=無効)       |
| FpuReduction  | 0.25                    | 未訪問の子のQ補正(0で無効)              |
| GPUId         | 0                       | 対局に使うGPU番号(-1でCPU)           |
| BookFile      | weights/book.bin        | 定跡ブック(無い場合は自動スキップ。USIで相対パス可) |
| BookMoves     | 40                      | この手数までブックを引く(0で無効)           |
| USI_Ponder    | true                    | Ponder有効/無効                  |

### 機能メモ

- 詰み判定:ルートで詰み発見時は確実に着手し、リーフで相手の詰みを検知して評価へ反映。npsを約25%消費するので不要ならMateDepth 0
- ツリー再利用:連続するgoで前回木を引き継ぐ(fo string tree reuse hit
- 持ち時間制御:go btime/wtime/binc/winc/byoyomi/movetime/nodes/infinite残り時間から1手の思考時間を自動配分(softで安定打ち切り/不安定ならhardまで延長)。
- Ponder対応: bestmove X ponder Yを返し、ponderperhittp。
- 安定性: 思考中の isreadyは再初期化不要なら即 adyokっている探索を壊さない)、
  ewgameositnをしてから状態を切り替える。
  内部例外はtminate即= 時間負け) にならずeesignio strinRROR ...をログに残す。

### 手動動作確認 (GUI を使わない場合)

(事前に weights/へ重みを配置しておくこと。ModelPath の値は自分の重みパスに合わせる)

```bash
( printf 'usi\nsetoption name ModelPath value weights/hisen_ema.pt\n'; \
  printf 'isready\nposition startpos\ngo movetime 1000\n'; \
  sleep 10; printf 'quit\n' ) | build/hisen-usi
# → info string loaded model: ... / info time ... nodes ... nps ... / bestmove 2g2f ...
```

(参考: RTX 4070 SUPER / Ryzen 9 7900X で Threads 16 時に nps 約 3000〜5000。
MCTS の1ノード = 30M パラメータNN1 回 forward のため、NNUE 系エンジンの nps とは直接比較できない。)
