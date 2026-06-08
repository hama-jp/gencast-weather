# GenCast Weather — ローカル気象予報アプリ

Google DeepMind の [GenCast](https://github.com/google-deepmind/graphcast) を使って、ローカルマシン上でアンサンブル気象予報を実行し、ブラウザで可視化するウェブアプリです。

## 特徴

- **GenCast 1p0deg Mini** モデルによる +12h アンサンブル予報
- 2m気温 / 海面気圧 / 風速 / アンサンブルスプレッドをLeaflet地図上に表示
- FastAPI バックグラウンド推論 + ブラウザでリアルタイム進捗確認
- 全ローカル完結

## 必要環境

| 項目 | 要件 |
|------|------|
| GPU | NVIDIA GPU (CUDA 12.x) |
| Python | 3.12 |
| RAM | 32GB 以上推奨 |

## セットアップ

```bash
# 1. 仮想環境
python3 -m venv .venv && source .venv/bin/activate

# 2. JAX (CUDA 12)
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. その他依存
pip install -r requirements.txt

# 4. モデル + データ ダウンロード（初回のみ、~300MB）
python scripts/download_data.py
```

## 起動

```bash
source .venv/bin/activate
cd app
XLA_PYTHON_CLIENT_PREALLOCATE=false python server.py
# → http://localhost:8000
```

## 技術スタック

- Google DeepMind GenCast (JAX + dm-haiku)
- FastAPI + uvicorn
- Leaflet.js + Canvas overlay

## ライセンス

- アプリコード: MIT
- GenCast: Apache 2.0 (Google DeepMind)
- ERA5: Copernicus CDS 利用規約
