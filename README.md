# Automatic detection of ring patterns based on deep learning
# ring_detr_latest

FM-AFM のエネルギー散逸像などに現れる **リング（楕円）パターン**を、DETR（Hugging Face `facebook/detr-resnet-50`）ベースで検出し、  
リングの幾何パラメータ（中心・長短軸・角度）を推定するためのリポジトリです。

> 目的：画像からリング候補を自動検出し、後段の R・SNR・統計解析（最近傍距離など）へつなげる。

---

## 特徴

- **RingDETR_HF**：Hugging Face DETR を backbone にしたカスタムモデル
  - 分類：`ring` vs `no-object`
  - 回帰：楕円パラメータ **[cx, cy, a, b, angle]**（+ optional thickness を扱う場合あり）
- **Hungarian matching**（線形割当）に基づく SetCriterion で学習
- **合成データ生成**（リング/楕円/厚み/ノイズなど）を同梱（Notebook内実装）
- データ形式は軽量：**images + labels.jsonl**

---

## リポジトリの想定構成（例）

```
ring_detr_latest/
  README.md
  example.ipynb              # 学習/推論/可視化の例（このリポジトリの主な入口）
  (optional) train.py         # notebook から切り出す場合
  (optional) infer.py
  data/
    train/
      images/
      labels.jsonl
    val/
      images/
      labels.jsonl
    test/
      images/
      labels.jsonl
```

> 実際のスクリプト構成はリポジトリの中身に合わせて適宜調整してください。  
> 現状、Notebook（`example.ipynb`）が “正” の実装になっている想定です。

---

## 必要環境

- Python 3.x
- PyTorch（CUDA 推奨）
- transformers（DETR）
- numpy / scipy / opencv-python / matplotlib / pandas

例：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy scipy opencv-python matplotlib pandas
```

---

## データ形式（必須）

### 1) 画像
- `data/<split>/images/` に配置
- グレースケール画像でもOK（内部で 3ch に複製して正規化する想定）
- 例：PNG / JPG など OpenCVで読める形式

### 2) ラベル（JSONL）
- `data/<split>/labels.jsonl`
- 1行 = 1画像のアノテーション

**例（1行）**
```json
{"file":"000001.png","params":[[cx,cy,a,b,angle,th], [cx,cy,a,b,angle,th]]}
```

- `file`：画像ファイル名（`images/` からの相対）
- `params`：リング（楕円）ごとのパラメータ配列
  - 1リング = `[cx, cy, a, b, angle, th]`
  - 学習で `th`（厚み）を使わない場合は、実装側で **6次元→5次元に切り落とし**ます（Notebook内の Dataset がそうなっています）
  - `params=[]`（リング無し画像）も許容

> Notebook内の `RingFolderDataset` は `labels` を全て 0（ring クラス）にしています。  
> no-object は DETR の “背景クラス” として内部処理されます。

---

## クイックスタート（Notebook）

### 1) Jupyter を起動
```bash
jupyter lab
# or
jupyter notebook
```

### 2) `example.ipynb` を上から順に実行
Notebookには通常、以下が含まれます：
- GPU / CUDA チェック
- `RingDETR_HF` 定義
- Dataset / DataLoader
- Hungarian matcher + loss（SetCriterion）
- 学習ループ（保存：`best.pt`, `last.pt` など）
- 推論と可視化

> まずは **data/ の配置**を合わせてから実行してください。

---

## 代表的な設定パラメータ（Notebook側で調整）

- `num_queries`：DETRのクエリ数（例：100）
- 入力サイズ：`resize_to=(512,512)` など
- 学習率 / batch size / epoch
- 目的関数の重み（分類/回帰）
- 合成データ生成のリング密度、半径分布、厚み分布、ノイズ量 など

---

## 推論の流れ（概略）

1. 画像読込（grayscale → 3ch）
2. DETR forward
3. `pred_logits` → スコア
4. `pred_params` → `[cx, cy, a, b, angle]`
5. スコア閾値 + NMS（必要なら）で絞り込み
6. 楕円を重ね描画して確認

---

## よくある注意

- **データの座標系**（cx,cy,a,b の単位）が学習・評価で一致しているか確認してください  
  - pixel 単位なのか、画像サイズで正規化（0–1）なのか
- 角度 `angle` の範囲（例：[-pi/2, pi/2] or [0,pi]）も統一
- DETR は学習初期に不安定になりやすいので、ログ（loss, score 分布）を見ながら調整推奨

---

## TODO（おすすめ改善）
- Notebook の処理を `train.py / infer.py` に切り出して CLI 化
- COCO形式への変換（外部ツールとの連携が楽）
- 合成データ生成器のパラメータを YAML/JSON で管理
- しきい値最適化（P_det, SNR 解析）を別モジュール化

---

## ライセンス / 引用
- DETR: Hugging Face / Facebook Research のモデル・重みを利用します（利用条件は各配布元に従ってください）
