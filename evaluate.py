# evaluate_ragas_table.py
# Danh gia chatbot RAG cua ban bang RAGAS tren API thuc te
# Yeu cau: /ask tra ve {"response": str, "contexts": List[str]}

import json
import sys
from typing import Any, Dict, List

import pandas as pd
import requests
from datasets import Dataset

import config
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_URL = "http://localhost:8000/ask_context"
EVAL_FILE = "evaluation_set.json"


def _normalize_contexts(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    text = str(raw).strip()
    return [text] if text else []


def _format_float(value: Any) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.3f}"


def _build_summary_frame(result: Any, sample_count: int) -> pd.DataFrame:
    summary: Dict[str, Any] = {}

    repr_dict = getattr(result, "_repr_dict", None)
    if isinstance(repr_dict, dict):
        summary.update(repr_dict)

    if not summary and hasattr(result, "scores"):
        try:
            scores_df = pd.DataFrame(result.scores)
        except Exception:
            scores_df = None
        if scores_df is not None and not scores_df.empty:
            numeric_df = scores_df.select_dtypes(include="number")
            if not numeric_df.empty:
                mean_series = numeric_df.mean(numeric_only=True)
                summary.update(mean_series.to_dict())

    if not summary and isinstance(result, dict):
        summary.update({k: v for k, v in result.items() if isinstance(v, (int, float))})

    if not summary:
        return pd.DataFrame()

    df = pd.DataFrame([summary])
    df.insert(0, "num_samples", sample_count)
    return df


def _prepare_run_config(sample_count: int) -> RunConfig:
    timeout_cfg = getattr(config, "RAGAS_TIMEOUT", 600)
    max_retries_cfg = getattr(config, "RAGAS_MAX_RETRIES", 5)
    max_workers_cfg = getattr(config, "RAGAS_MAX_WORKERS", 1)
    log_tenacity_cfg = getattr(config, "RAGAS_LOG_TENACITY", False)

    try:
        timeout = int(timeout_cfg)
    except (TypeError, ValueError):
        timeout = 600

    try:
        max_retries = int(max_retries_cfg)
    except (TypeError, ValueError):
        max_retries = 4

    try:
        max_workers = int(max_workers_cfg)
    except (TypeError, ValueError):
        max_workers = 4

    max_workers = max(1, min(max_workers, max(1, sample_count)))

    return RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_workers=max_workers,
        log_tenacity=bool(log_tenacity_cfg),
    )


def main() -> None:
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    records: List[Dict[str, Any]] = []
    for index, sample in enumerate(eval_samples, start=1):
        question = sample.get("question", "")
        try:
            response = requests.post(API_URL, json={"text": question}, timeout=300)
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "")
            contexts = _normalize_contexts(data.get("contexts"))
        except Exception as exc:
            print(f"[{index}] Loi goi API: {exc}")
            answer = ""
            contexts = []

        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample.get("ground_truth", ""),
            }
        )

    dataset = Dataset.from_list(records)

    ollama_base_url = getattr(config, "OLLAMA_API_URL", "http://localhost:11434")
    if "/api" in ollama_base_url:
        ollama_base_url = ollama_base_url.split("/api", 1)[0]

    ragas_llm = LangchainLLMWrapper(
        ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=ollama_base_url,
            temperature=0.0,
        )
    )
    embedding_device = getattr(config, "EMBEDDING_DEVICE", "cpu")
    langchain_embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = _prepare_run_config(sample_count=len(records))

    batch_size_cfg = getattr(config, "RAGAS_BATCH_SIZE", None)
    try:
        batch_size = int(batch_size_cfg) if batch_size_cfg is not None else None
    except (TypeError, ValueError):
        batch_size = None

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        batch_size=batch_size,
    )

    summary_df = _build_summary_frame(result, sample_count=len(records))

    print("\nBANG KET QUA RAGAS (Pipeline: CURRENT /ask)\n")
    if summary_df.empty:
        print("Khong the tinh toan diem trung binh. Kiem tra lai log khong loi.")
    else:
        numeric_cols = summary_df.select_dtypes(include="number").columns
        formatters = {col: (lambda value, _col=col: _format_float(value)) for col in numeric_cols}
        print(summary_df.to_string(index=False, formatters=formatters))
        summary_df.to_csv("ragas_results.csv", index=False)
        print("\nDa luu: ragas_results.csv")

    per_sample_df = None
    try:
        per_sample_df = result.to_pandas()
    except Exception:
        per_sample_df = None

    if per_sample_df is not None and not per_sample_df.empty:
        if "contexts" in per_sample_df.columns:
            per_sample_df["contexts"] = per_sample_df["contexts"].apply(
                lambda value: " || ".join(value) if isinstance(value, list) else value
            )
        per_sample_df.to_csv("ragas_per_sample.csv", index=False)
        print("Da luu chi tiet mau: ragas_per_sample.csv")


if __name__ == "__main__":
    main()
