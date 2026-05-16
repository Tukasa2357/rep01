import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Optional

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

CONFIG_PATH = Path("settings.json")
MODEL_OPTIONS = ["gpt-5.3-codex", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
DEFAULT_MODEL = MODEL_OPTIONS[0]


SECTION_HINTS = [
    "要約",
    "請求項",
    "特許請求の範囲",
    "発明の詳細な説明",
    "解決しようとする課題",
    "課題を解決するための手段",
    "発明の効果",
]


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


def _extract_pages_text(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return pages


def _rank_page_score(text: str) -> int:
    score = 0
    for hint in SECTION_HINTS:
        if hint in text:
            score += 4
    score += min(len(text) // 300, 10)
    return score


def extract_text_from_pdf(pdf_bytes: bytes, max_chars: int = 24000) -> str:
    pages = _extract_pages_text(pdf_bytes)
    indexed = [(idx, text, _rank_page_score(text)) for idx, text in enumerate(pages) if text]
    # 重要度順（同点なら前ページ優先）
    indexed.sort(key=lambda x: (-x[2], x[0]))

    chunks: list[str] = []
    total = 0
    for page_idx, page_text, _ in indexed:
        remaining = max_chars - total
        if remaining <= 0:
            break
        clipped = page_text[:remaining]
        chunks.append(f"[Page {page_idx + 1}]\n{clipped}")
        total += len(clipped)

    return "\n\n".join(chunks)


def ocr_fallback_with_llm(
    *,
    pdf_bytes: bytes,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> str:
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "このPDFが画像ベースである可能性があります。"
                            "OCRとして本文をできるだけ忠実に日本語で書き起こしてください。"
                            "ページ番号を付けて出力してください。"
                        ),
                    },
                    {
                        "type": "input_file",
                        "filename": "patent.pdf",
                        "file_data": f"data:application/pdf;base64,{b64}",
                    },
                ],
            }
        ],
    )
    return response.output_text


def summarize_patent(
    *,
    text: str,
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
) -> dict:
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    page_summary_prompt = (
        "以下の特許公報テキストを読み、重要ポイントのみJSONで出力してください。"
        "キーは overview/problem/configuration/effect/use_cases/evidence に限定し、"
        "各値は日本語の短い箇条書き配列にしてください。"
    )

    page_summary = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "JSONのみ出力。事実に忠実。"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": page_summary_prompt},
                    {"type": "input_text", "text": text},
                ],
            },
        ],
    ).output_text

    synth_prompt = (
        "次の中間JSONを統合し、最終要約JSONを出力してください。"
        "キー: overview/problem/configuration/effect/use_cases/quality_checks。"
        "quality_checks には faithful_to_source, missing_risk, notes を含めること。"
    )

    final_summary_text = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "JSONのみ出力。推測は最小化。"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": synth_prompt},
                    {"type": "input_text", "text": page_summary},
                ],
            },
        ],
    ).output_text

    try:
        return json.loads(final_summary_text)
    except json.JSONDecodeError:
        return {
            "overview": [],
            "problem": [],
            "configuration": [],
            "effect": [],
            "use_cases": [],
            "quality_checks": {
                "faithful_to_source": "unknown",
                "missing_risk": "unknown",
                "notes": ["JSONとして解析できない形式で返却されました。"],
            },
            "raw_output": final_summary_text,
        }


def main() -> None:
    st.set_page_config(page_title="特許公報 要約ツール", layout="wide")
    st.title("特許公報PDF 要約ツール")

    settings = load_settings()
    saved_model = settings.get("model", DEFAULT_MODEL)
    model_index = MODEL_OPTIONS.index(saved_model) if saved_model in MODEL_OPTIONS else 0

    with st.sidebar:
        st.header("設定")
        api_key = st.text_input("OpenAI API Key", value=settings.get("api_key", ""), type="password")
        model = st.selectbox("モデル名", options=MODEL_OPTIONS, index=model_index)
        base_url = st.text_input("Base URL（任意）", value=settings.get("base_url", ""))
        use_ocr_fallback = st.checkbox("抽出失敗時にLLM OCRフォールバックを使う", value=True)

        if st.button("設定を保存"):
            save_settings(api_key=api_key, model=model, base_url=base_url)
            st.success("設定を保存しました。")

    st.write("PDFをファイル選択ダイアログから選んで、特許公報の要約を生成します。")
    uploaded_file = st.file_uploader("PDFファイルを選択", type=["pdf"])

    if st.button("要約を作成"):
        if not api_key:
            st.error("API Keyを設定してください。")
            return

        if uploaded_file is None:
            st.error("PDFファイルを選択してください。")
            return

        pdf_bytes = uploaded_file.getvalue()

        with st.spinner("PDF読込中..."):
            text = extract_text_from_pdf(pdf_bytes)

        if len(text.strip()) < 500 and use_ocr_fallback:
            with st.spinner("通常抽出が不足のため、LLM OCRを実行中..."):
                try:
                    text = ocr_fallback_with_llm(
                        pdf_bytes=pdf_bytes,
                        api_key=api_key,
                        model=model,
                        base_url=base_url or None,
                    )
                except Exception as exc:
                    st.warning("OCRフォールバックに失敗しました。通常抽出結果で続行します。")
                    st.caption(str(exc))

        if not text.strip():
            st.error("PDFから本文を抽出できませんでした。")
            return

        with st.expander("抽出テキスト（先頭2000文字）"):
            st.text(text[:2000])

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
        st.json(summary)
        st.download_button(
            "要約JSONをダウンロード",
            data=json.dumps(summary, ensure_ascii=False, indent=2),
            file_name="patent_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
