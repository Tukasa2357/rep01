# 特許公報PDF要約ツール

ユーザが指定した特許公報のPDFを読み込み、ChatGPT APIで要約を生成するStreamlitアプリです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動

```bash
streamlit run app.py
```

## 使い方

1. サイドバーの「設定」で `OpenAI API Key` を入力し、モデルをプルダウンから選択して保存。
2. 「PDFファイルを選択」から対象PDFを選ぶ。
3. 「要約を作成」を押すと、抽出本文をもとに要約が表示されます。

設定は `settings.json` に保存されます。
