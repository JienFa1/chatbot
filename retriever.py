# retriever.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Optional, Dict, Any, Collection
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pyvi import ViTokenizer
from collections import defaultdict
import json
import re
import requests
import config
import torch

torch.cuda.empty_cache()

VI_STOP_PHRASES = {p.replace(" ", "_") for p in config.VI_STOP_PHRASES}
DEFAULT_STOPWORDS = (
    set(map(str.lower, config.VI_STOPWORDS))
    .union(VI_STOP_PHRASES)
)
def _vi_preprocess(text: str) -> str:
    TOKEN_RE = re.compile(r"^[a-zA-ZÀ-ỹ_]+$", re.UNICODE)
    s = ViTokenizer.tokenize(str(text).lower())
    toks = []
    for t in s.split():
        if t in DEFAULT_STOPWORDS:
            continue
        if not TOKEN_RE.match(t):
            continue
        toks.append(t)
    return " ".join(toks)

def _extract_prf_terms(docs: List[Document], top_m: int = 6) -> List[str]:
    texts = [d.page_content for d in docs if getattr(d, "page_content", None)]
    stop_words = list(DEFAULT_STOPWORDS)
    if not texts:
        return []
    vec = TfidfVectorizer(tokenizer=_vi_preprocess,  token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), 
                          stop_words=stop_words, strip_accents=None, max_features=4096)
    X = vec.fit_transform(texts)                # shape: (#docs, #terms)
    scores = np.asarray(X.sum(axis=0)).ravel()  # tổng tf-idf theo term
    terms = np.array(vec.get_feature_names_out())
    idx = scores.argsort()[::-1][:top_m]
    return terms[idx].tolist()
def _mean_vector(vectors: List[List[float]]) -> Optional[List[float]]:
    if not vectors:
        return None
    return [float(sum(col)) / len(vectors) for col in zip(*vectors)]

def _dedup(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source") or d.metadata.get("url"), d.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def _reciprocal_rank_fusion(results, k=60, return_scores=False, key_fn=None):
   
    if key_fn is None:
        def key_fn(doc):
            if isinstance(doc, Document):
                src = doc.metadata.get("source", "")
                page = doc.metadata.get("page", "")
                return json.dumps([doc.page_content, src, page], ensure_ascii=False)
            try:
                return json.dumps(doc, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(doc)

    fused_scores = defaultdict(float)
    keep_doc = {}

    for docs in results:                      # results: list of ranked lists
        for rank, doc in enumerate(docs):
            key = key_fn(doc)
            keep_doc[key] = doc               
            fused_scores[key] += 1.0 / (rank + k)  # RRF

    items = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

    if return_scores:
        return [(keep_doc[k], score) for k, score in items]
    else:
        return [keep_doc[k] for k, _ in items]

class Retriever:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size" : 64 }
        )
        self.embeddings = embeddings
        self.vectorstore = FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        self._all_docs: List[Document] = list(getattr(self.vectorstore.docstore, "_dict", {}).values())
        self.dense_baseline = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.4},
        )
        self.bm25 = BM25Retriever.from_documents(self._all_docs, preprocess_func = _vi_preprocess) if self._all_docs else None
        if self.bm25:
            self.bm25.k = 10
        try:
            self.cross_encoder = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", device="cuda", automodel_args={"torch_dtype": "float16"})
        except Exception:
            self.cross_encoder = None

# ---------- LLM Multi-query (qua Ollama API trá»±c tiáº¿p, KHÃ”NG dÃ¹ng OllamaInterface) ----------
    def _llm_generate_queries(self, question: str) -> List[str]:
        prompt = (
            "bạn là 1 trợ lý AI thông minh.\n"
            "Hãy tạo cho tôi 3 phiên bản câu hỏi khác từ câu hỏi gốc\n"
            "Bằng cách tạo nhiều câu hỏi mang các góc nhìn (perspective) khác nhau từ câu hỏi gốc của người dùng, mục tiêu của bạn là giúp người dùng vượt qua những giới hạn về distance-based similarity search.\n"
            "Yêu cầu:\n"
            f"Chỉ in ra {config.QUERY_GEN_NUM} câu truy vấn mới: không đánh số; không ngoặc kép; mỗi truy vấn 1 dòng."
            "Chỉ liệt kê câu hỏi, không viết câu giới thiệu"
            f"Câu hỏi gốc: {question}\n"
        )
        payload = {
            "model": getattr(config, "QUERY_GEN_MODEL", config.OLLAMA_MODEL),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9}
        }
        try:
            r = requests.post(config.OLLAMA_API_URL, json=payload, timeout=60)
            r.raise_for_status()
            text = r.json().get("response", "")
            lines = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                s = re.sub(r'^\s*[\-\*\d]+[.)]?\s*', '', s)  # "- ", "1) ", "1. "
                s = s.strip(' "\'')
                if s:
                    lines.append(s)
            lines = lines[: int(getattr(config, "QUERY_GEN_NUM", 2))]
            return [question] + lines if lines else [question]
        except Exception:
            return [question]
    
    def _hybrid_retrieve_prf(self,query: str,*,k: int = 10,fb_docs: int = 8,alpha: float = 0.6,prf_terms: int = 6,
    ) -> List[List[Document]]:
        dense_1: List[Document] = self.dense_baseline.invoke(query) if self.dense_baseline else []
        sparse_1: List[Document] = (self.bm25.invoke(query) if self.bm25 else [])
        
        fb: List[Document] = []
        fb += dense_1[:max(1, fb_docs // 2)]
        fb += sparse_1[:max(1, fb_docs - len(fb))]
        if not fb:
            return [_dedup(dense_1)[:k], _dedup(sparse_1)[:k]]
        
        # Sparse PRF: TF-IDF terms
        expansion = _extract_prf_terms(fb, top_m=prf_terms)
        expanded_query = query if not expansion else (query + " " + " ".join(expansion))
        
        # Dense PRF: 
        fb_texts = [d.page_content for d in fb if getattr(d, "page_content", None)]
        try:
            q_vec = self.embeddings.embed_query(query)
            if fb_texts:
                fb_vecs = self.embeddings.embed_documents(fb_texts)
                m_vec = _mean_vector(fb_vecs)
                prf_vec = [(1 - alpha) * q + alpha * m for q, m in zip(q_vec, m_vec)]
                prf_vec = np.array(prf_vec, dtype=np.float32)
                prf_vec = prf_vec / (np.linalg.norm(prf_vec) + 1e-12)
                dense_2 = self.vectorstore.similarity_search_by_vector(prf_vec, k=max(k, 20))
            else:
                dense_2 = dense_1[:k]
        except Exception:
            dense_2 = dense_1[:k]
        
        if self.bm25:
            old_k = getattr(self.bm25, "k", None)
            self.bm25.k = max(k, 20) 
            sparse_2 = self.bm25.invoke(expanded_query)
            if old_k is not None:
                self.bm25.k = old_k
        else:
            sparse_2 = sparse_1[:k]
        return [_dedup(dense_2)[:max(k, 20)], _dedup(sparse_2)[:max(k, 20)]]
    
    # ---------- RRF across queries ----------
    def _retrieve_rrf_for_queries(self, queries: List[str], rrf_k: int = 60) -> List[Document]:
        per_query_rrf: List[List[Document]] = []
        for q in queries:
            dense_and_sparse = self._hybrid_retrieve_prf(q)
            fused_per_query = _reciprocal_rank_fusion(dense_and_sparse, k=rrf_k)
            per_query_rrf.append(fused_per_query)
        return _reciprocal_rank_fusion(per_query_rrf, k=rrf_k)
    
    # ---------- Cross-encoder rerank ----------
    def _cross_rerank(self, query: str, docs: List[Document], top_k: int = 8, depth:int = 40) -> List[Document]:
        if not docs:
            return []
        if self.cross_encoder is None:
            return docs[:top_k]
        
        candidates = docs[:min(depth, len(docs))] 
        pairs = [(query, d.page_content) for d in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]
    
    def get_relevant_chunks(self, question: str, k: int = 8) -> List[str]:
        queries = self._llm_generate_queries(question)

        pool_docs = self._retrieve_rrf_for_queries(queries, rrf_k=60)

        depth = min(40, len(pool_docs))
        top_docs = self._cross_rerank(question, pool_docs, top_k=k, depth=depth)

        return [d.page_content for d in top_docs]
