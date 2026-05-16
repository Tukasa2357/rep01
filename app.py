import json
from pathlib import Path
from typing import Optional

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

CONFIG_PATH = Path("settings.json")
DEFAULT_MODEL = "gpt-4.1-mini"


def load_settings() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_settings(api_key: str, model: str, base_url: str) -> None:
    payload = {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
    }
    CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 24000) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    total = 0
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        clipped = page_text[:remaining]
        chunks.append(clipped)
        total += len(clipped)
    return "\n\n".join(chunks)


def summarize_patent(
    *,
    text: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> str:
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    prompt = (
        "あなたは特許調査の専門家です。以下の特許公報本文を読み、"
        "次の見出しで日本語要約を作成してください。\n"
        "1. 発明の概要\n"
        "2. 解決しようとする課題\n"
        "3. 主要な構成\n"
        "4. 期待される効果\n"
        "5. 想定される用途\n"
        "各項目は簡潔に箇条書きでまとめてください。"
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "正確で簡潔に要約してください。"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_text", "text": text},
                ],
            },
        ],
    )
    return response.output_text


def main() -> None:
    st.set_page_config(page_title="特許公報 要約ツール", layout="wide")
    st.title("特許公報PDF 要約ツール")

    settings = load_settings()

    with st.sidebar:
        st.header("設定")
        api_key = st.text_input("OpenAI API Key", value=settings.get("api_key", ""), type="password")
        model = st.text_input("モデル名", value=settings.get("model", DEFAULT_MODEL))
        base_url = st.text_input("Base URL（任意）", value=settings.get("base_url", ""))

        if st.button("設定を保存"):
            save_settings(api_key=api_key, model=model, base_url=base_url)
            st.success("設定を保存しました。")

    st.write("PDFを指定して、特許公報の要約を生成します。")
    pdf_path_str = st.text_input("PDFファイルパス", value="")

    if st.button("要約を作成"):
        if not api_key:
            st.error("API Keyを設定してください。")
            return

        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            st.error(f"PDFが見つかりません: {pdf_path}")
            return

        with st.spinner("PDF読込中..."):
            text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            st.error("PDFから本文を抽出できませんでした。")
            return

        with st.spinner("ChatGPTで要約作成中..."):
            try:
                summary = summarize_patent(
                    text=text,
                    api_key=api_key,
                    model=model,
                    base_url=base_url or None,
                )
            except Exception as exc:
                st.exception(exc)
                return

        st.subheader("要約結果")
        st.write(summary)


if __name__ == "__main__":
    main()
