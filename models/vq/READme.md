# VQディレクトリについて
## model.pyについて  
encoderが最初に受ける入力は `30*128` に設定されている。これは「30部位 × 各部位の128次元」の平坦化された空間埋め込みを想定した値です。

## モデルの全体像

このディレクトリには、Residual Vector-Quantized VAE（RVQVAE）と関連モジュールの実装が含まれます。主要なクラスと役割は以下の通りです。

- `models/vq/model.py`:
  - `RVQVAE`：Encoder → ResidualVQ（複数量子化）→ Decoder の流れでモーション系列を圧縮・再構成する条件付きモデル（`species`, `gender` を条件として使用）。
  - `LengthEstimator`：テキスト埋め込みなどから生成系列長を予測する小さな MLP。
- `models/vq/encdec.py`：`Encoder` / `Decoder` / `FiLMLayer` / `Spatial_Transformer` の実装。Spatial（関節間）トランスフォーマ → 時系列 Conv/ResNet ブロック のハイブリッド構成。
- `models/vq/residual_vq.py`：`ResidualVQ` の実装。複数段の量子化器を用いて残差ベースに表現を量子化する。
- `models/vq/quantizer.py`：個々の量子化器（`QuantizeEMAReset` など）の具体実装（EMA 更新等）。

## 層構成（大枠）

RVQVAE の層的構成は次のようになっています。

1. 前処理 / 後処理
	- 入力は (batch, T, D)（例: D=359）で渡され、`preprocess` により (batch, D, T) に permute される。
2. Encoder
	- 各関節の特徴を線形層で埋め込み（root, other joints, contact 等）。
	- `Spatial_Transformer` により関節（空間）方向の文脈を獲得し、各フレームで `30 × 128` の埋め込みにする。
	- 埋め込みを (batch, 30*128, T) として Conv1d / Resnet1D に通し、時間方向にダウンサンプリングして最終的に (batch, output_emb_width, T') を出力。
	- FiLM 層でテキスト条件（CLIP 埋め込み）を注入する。
3. ResidualVQ
	- `num_quantizers` 個の量子化層を残差的に適用。各層は量子化埋め込み・インデックス・ロス・perplexity を返す。
	- 出力は量子化された連続埋め込み `quantized_out`（Decoder への入力）と、コードインデックス `all_indices`。
4. Decoder
	- (batch, code_dim, T') を受け取り Resnet1D + Upsample により時間解像度を復元、最終的に (batch, T, input_width) の再構成を返す。

## 代表的なテンソル形状（デフォルト設定での例）

- 元入力: (B, T, D) 例: D = 359
- preprocess -> (B, D, T)
- Spatial Transformer 出力 -> (B, T, 30*128) = (B, T, 3840)
- permute -> (B, 3840, T) （これが Encoder の最初の Conv1d の in_channels）
- downsampling (down_t=3, stride_t=2) により時間長は約 T/8 になる（パディングの扱いに依存）
- Encoder 出力 -> (B, 512, T')（= (B, code_dim, T')）
- ResidualVQ 出力 `quantized_out` -> (B, 512, T')
- Decoder 出力 -> (B, T, input_width) 例: (B, T, 359)

## 主要な使用箇所

- 学習: `train_vq.py`（VQ の単体学習）
- 条件付き生成/トランスフォーマ学習: `train_t2m_transformer.py`, `train_res_transformer.py`（事前に学習済みの VQ をロードし、コード列をトークンとして扱う）
- 評価: `utils/eval_t2m.py`（`encode` / `forward_decoder` を用いて生成を評価）

## よくある注意点

- `30*128` は `encdec.Encoder` の設計（Spatial Transformer の出力）に依存するため、関節数や per-joint 埋め込み次元を変更する場合は `Encoder` 側とここを一緒に変更する必要があります。
- `output_emb_width`（Encoder の出力幅）と `code_dim`（量子化ベクトル次元）は一致させる必要があります（`model.py` に assert がある）。
- downsampling / upsampling の整合性（時間長の端数）に注意。round-trip（入力→encode→decoder）で形状と数値的整合を確認すること。

## 変更方法のヒント

- `30`（部位数）や `128`（各部位の埋め込み次元）を可変にしたい場合は、`RVQVAE` のコンストラクタに `num_parts` と `per_part_dim` の引数を追加し、`Encoder(num_parts * per_part_dim, ...)` のように動的に渡すと良いです。
- 例えば:

```python
# model.py のコンストラクタ例（抜粋）
num_parts = 30
per_part_dim = 128
self.encoder = Encoder(num_parts * per_part_dim, output_emb_width, ...)
```
## 自分の解説
- 
## 参考ファイル

- `models/vq/encdec.py` — Encoder/Decoder の実装
- `models/vq/residual_vq.py` — ResidualVQ の実装
- `models/vq/quantizer.py` — 個別量子化器の実装
- `train_vq.py`, `train_t2m_transformer.py`, `utils/eval_t2m.py` — 使用例・評価パイプライン

---