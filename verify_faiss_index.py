import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

LOG_FILE = Path("faiss_verification.log")


def configure_logger() -> None:
    handlers = [
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def ensure_index_artifacts(index_path: Path) -> None:
    if not index_path.exists():
        logging.error("Index directory %s does not exist.", index_path)
        return

    expected_files = {"index.faiss", "index.pkl"}
    available = {p.name for p in index_path.iterdir() if p.is_file()}
    missing = expected_files - available
    if missing:
        logging.warning("Index directory is missing files: %s", ", ".join(sorted(missing)))
    else:
        logging.info("Found FAISS artifacts: %s", ", ".join(sorted(expected_files)))


def load_vectorstore() -> Optional[FAISS]:
    try:
        device = getattr(config, "EMBEDDING_DEVICE", "cpu")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logging.info("Loading FAISS index from %s", config.FAISS_INDEX_PATH)
        vectorstore = FAISS.load_local(
            config.FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception as exc:
        logging.exception("Failed to load FAISS index: %s", exc)
        return None


def _format_vector(raw_vector: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in raw_vector) + "]"


def _resolve_id_entries(vectorstore: FAISS) -> List[Tuple[int, str]]:
    raw_id_map = getattr(vectorstore, "index_to_docstore_id", {})
    if isinstance(raw_id_map, dict):
        # entries like {0: "uuid"}
        return [(int(idx), doc_id) for idx, doc_id in raw_id_map.items()]
    # fallback: treat as sequence of doc IDs ordered by position
    sequence = list(raw_id_map)
    return list(enumerate(sequence))


def log_samples(vectorstore: FAISS, preview: int = 5) -> None:
    faiss_index = getattr(vectorstore, "index", None)
    if faiss_index is None or not hasattr(faiss_index, "reconstruct"):
        logging.error("FAISS index object does not expose reconstruct().")
        return

    docstore = getattr(vectorstore, "docstore", None)
    raw_docs: Dict[str, Any] = getattr(docstore, "_dict", {}) if docstore else {}
    id_entries = _resolve_id_entries(vectorstore)

    if not raw_docs or not id_entries:
        logging.warning("No documents found in the index.")
        return

    logging.info("Previewing source/text/vector for %s entries.", min(preview, len(id_entries)))

    for position, doc_id in id_entries[:preview]:
        doc = raw_docs.get(doc_id)
        if doc is None and isinstance(doc_id, str):
            doc = raw_docs.get(doc_id.strip())
        if doc is None:
            logging.warning("Missing docstore entry for doc_id=%s", doc_id)
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source") or "<unknown>"
        text = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        if len(text) > 500:
            text = text[:500] + "..."

        try:
            vector = faiss_index.reconstruct(int(position))
            vector_str = _format_vector(vector)
        except Exception as exc:
            logging.error("Unable to reconstruct vector at position %s: %s", position, exc)
            continue

        logging.info("Source: %s", source)
        logging.info("Text: %s", text)
        logging.info("Vector: %s", vector_str)


def main() -> None:
    configure_logger()

    index_path = Path(config.FAISS_INDEX_PATH)
    logging.info("Verifying FAISS index under %s", index_path.resolve())
    ensure_index_artifacts(index_path)

    vectorstore = load_vectorstore()
    if vectorstore is None:
        logging.error("Stopping verification because the vectorstore could not be loaded.")
        return

    log_samples(vectorstore)
    logging.info("Verification complete. Full log saved to %s", LOG_FILE.resolve())


if __name__ == "__main__":
    main()
