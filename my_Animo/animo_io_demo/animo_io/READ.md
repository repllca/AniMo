# AniMo IO Demo（AniMoLite 簡易版）

本リポジトリは、  
**テキスト（日本語・英語）から動物のスケルトンモーションを生成する  
AniMo 系 Text-to-Motion の「推論パイプライン」を最小構成で実装したデモ**です。

研究目的として、まず **入出力（I/O）と推論フローを完全に固定**し、  
内部アルゴリズム（学習・モデル構造）は後から自由に差し替えられる設計にしています。

---

## このリポジトリで「できること」

✅ テキスト（日本語 / 英語）を入力  
✅ AniMo 風の形式でモーションを **推論（inference）**  
✅ 動物スケルトンとして **3Dで可視化**  
✅ 学習なしでも動く（未学習モデルでの推論）  
✅ 後から AniMo / MoMask / RAC / 自己教師ありに拡張可能

---

## 現在の到達点（重要）

### ✔ 推論は **すでに可能**
- PyTorch モデル（AniMoLite）による forward 計算が動作
- パラメータ固定の状態で  
  **text → motion を生成 = 推論が成立**
- 同じ入力テキスト → 同じ出力（再現性あり）

※ 現時点では「学習済みモデル」ではありませんが、  
**研究・デモ文脈では完全に「推論フェーズ」**です。

---

## 全体の流れ（パイプライン）

テキスト（日本語 / 英語）
↓
Text Embedder（今は hash ベース）
↓
AniMoLite（Transformer による時系列生成）
↓
Skeleton + Motion（AniMoLikeOutput）
↓
Forward Kinematics（FK）
↓
3D スケルトンアニメーション表示

markdown
コードをコピーする

---

## 入出力仕様（I/O コントラクト）

### 入力
- `text: str`  
  - 日本語・英語どちらでも可  
  - 例: `"犬が歩く / a dog is walking"`
- `species: str`（今は `"dog"` 固定）
- `T: int`（フレーム数）

---

### 出力：`AniMoLikeOutput`

#### Skeleton（固定情報）
- `joint_names: list[str]`（関節名, 長さ = J）
- `parents: list[int]`（親関節 index, root は `-1`）
- `rest_offsets: np.ndarray`（shape = `[J,3]`）

#### Motion（時系列）
- `root_translation: np.ndarray`（shape = `[T,3]`）
- `joint_quat: np.ndarray`（shape = `[T,J,4]` quaternion）
- `foot_contacts: np.ndarray | None`（shape = `[T,F]`）

👉 **この形式は今後も不変**  
中身のモデルが変わっても、必ずこの形で出力します。

---

## ディレクトリ構成

animo_io/
├─ types.py # I/O仕様（最重要）
├─ embedding/ # text → embedding
│ ├─ base.py
│ └─ hash_embedder.py
├─ skeletons/ # species → skeleton
│ ├─ base.py
│ └─ toy_dog.py
├─ generators/ # text + skeleton → motion
│ ├─ base.py
│ ├─ dummy_generator.py # 学習不要の最小生成器
│ ├─ animo_lite_model.py # Transformerモデル
│ └─ animo_lite_generator.py
infer.py # 推論用エントリポイント
visualize_motion.py # FK + 3D可視化

yaml
コードをコピーする

---

## セットアップ

### 必要環境
- Python 3.7+
- numpy
- matplotlib
- torch

例：
```bash
pip install numpy matplotlib torch
※ projection="3d" エラー対策として
visualize_motion.py 内で mpl_toolkits.mplot3d を明示的に import しています。

推論の実行方法（重要）
1️⃣ 推論のみ（shape確認）
bash
コードをコピーする
python infer.py --text "犬が歩く / a dog is walking" --T 120 --device cpu
出力例：

makefile
コードをコピーする
J = 5
T = 120
root_translation: (120, 3)
joint_quat: (120, 5, 4)
foot_contacts: (120, 2)
2️⃣ 推論 + 3D可視化（デモ）
bash
コードをコピーする
python infer.py \
  --text "犬が歩く / a dog is walking" \
  --T 180 \
  --device cpu \
  --visualize
オプション：

--stride 2 : フレーム間引き（軽量化）

--fps 30 : 再生速度

--device cuda : GPU使用

--weights path/to/model.pt : 学習済み重みをロード（任意）

AniMoLite について（簡易版 AniMo）
TransformerEncoder を用いた 時系列生成モデル

テキスト埋め込み + 時刻埋め込み → モーション生成

出力を

root（平行移動）

joint（回転）

contact（接地）
に分けて予測

👉 本家 AniMo の設計思想を
最小構成で再現した「形を作るためのモデル」

研究的な位置づけ
現在：

推論パイプライン完成

デモとして「動く・見える・説明できる」状態

今後：

学習（loss 定義・データセット接続）

MoMask / RAC への差し替え

自己教師あり（動物動画）への拡張

まとめ（現状を一言で）
Text-to-Motion（AniMo 系）の
推論フローと I/O を完全に固定したデモ基盤が完成している。

現在は未学習モデルだが、
学習済みモデルと同一の推論経路で動作する。

