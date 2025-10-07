import streamlit as st
import logging
import warnings

try:
    from langchain_core.exceptions import LangChainDeprecationWarning
except ImportError:  # pragma: no cover
    class LangChainDeprecationWarning(DeprecationWarning):
        pass

from chatbot import Chatbot
from retriever import _reciprocal_rank_fusion


st.set_page_config(page_title="Chatbot Q&A")
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens", category=UserWarning, module="sklearn.feature_extraction.text")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)



def instrument_retriever(retriever) -> None:
    if getattr(retriever, "_streamlit_instrumented", False):
        return

    original_llm_generate = retriever._llm_generate_queries

    def wrapped_llm_generate(question: str):
        queries = original_llm_generate(question)
        retriever._last_subqueries = queries
        return queries

    retriever._llm_generate_queries = wrapped_llm_generate

    original_rrf = retriever._retrieve_rrf_for_queries

    def wrapped_rrf(queries, rrf_k: int = 60):
        context_log = {}
        for q in queries:
            try:
                dense_and_sparse = retriever._hybrid_retrieve_prf(q)
                fused_docs = _reciprocal_rank_fusion(dense_and_sparse, k=rrf_k)
                snippets = []
                for doc in fused_docs[:5]:
                    prefix_parts = []
                    source = getattr(doc, "metadata", {}).get("source") if hasattr(doc, "metadata") else None
                    page = getattr(doc, "metadata", {}).get("page") if hasattr(doc, "metadata") else None
                    if source:
                        prefix_parts.append(str(source))
                    if page not in (None, ""):
                        prefix_parts.append(f"page {page}")
                    prefix = " - ".join(prefix_parts)
                    text = (doc.page_content or "").strip()
                    if prefix:
                        snippets.append(f"{prefix}: {text}")
                    else:
                        snippets.append(text)
                context_log[q] = snippets or ["(no context chunks retrieved)"]
            except Exception as exc:  # pragma: no cover - for robustness only
                context_log[q] = [f"Unable to capture context: {exc}"]
        retriever._last_subquery_contexts = context_log
        return original_rrf(queries, rrf_k=rrf_k)

    retriever._retrieve_rrf_for_queries = wrapped_rrf

    original_get_chunks = retriever.get_relevant_chunks

    def wrapped_get_chunks(question: str, k: int = 8):
        chunks = original_get_chunks(question, k=k)
        retriever._last_final_context = "\n".join(chunks)
        return chunks

    retriever.get_relevant_chunks = wrapped_get_chunks
    retriever._streamlit_instrumented = True



def render_log(log_data) -> None:
    if not log_data:
        return

    subqueries = log_data.get("subqueries") or []
    contexts = log_data.get("contexts") or {}
    final_context = log_data.get("final_context")

    with st.container():
        st.markdown("### Thinking")
        if subqueries:
            st.markdown("**Subqueries**")
            for idx, subq in enumerate(subqueries, start=1):
                st.markdown(f"{idx}. {subq}")
                snippets = contexts.get(subq) or []
                if snippets:
                    with st.expander(f"Context for subquery {idx}", expanded=False):
                        for chunk_idx, snippet in enumerate(snippets, start=1):
                            st.markdown(f"{chunk_idx}. {snippet}")
                else:
                    st.caption("No context chunks retrieved")
        if final_context:
            with st.expander("Final context sent to LLM", expanded=False):
                st.markdown(final_context)

@st.cache_resource(show_spinner=False)
def load_chatbot() -> Chatbot:
    chatbot = Chatbot()
    instrument_retriever(chatbot.retriever)
    return chatbot


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            log_data = message.get("log")
            if log_data:
                render_log(log_data)


def main() -> None:
    st.title("Chatbot question-answering")
    st.caption("Ask a question based on your indexed documents and get an answer from the chatbot.")

    chatbot = load_chatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.experimental_rerun()

    render_history()

    question = st.chat_input("Enter your question")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    assistant_message = {"role": "assistant"}

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                answer = chatbot.get_response(question)
            except Exception as exc:  # pragma: no cover - UI feedback only
                answer = f"Failed to generate answer: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

        subqueries = getattr(chatbot.retriever, "_last_subqueries", [])
        contexts = getattr(chatbot.retriever, "_last_subquery_contexts", {})
        final_context = getattr(chatbot.retriever, "_last_final_context", "")

        log_payload = {}
        if subqueries:
            log_payload["subqueries"] = subqueries
        if contexts:
            log_payload["contexts"] = contexts
        if final_context:
            log_payload["final_context"] = final_context

        assistant_message["content"] = answer
        if log_payload:
            assistant_message["log"] = log_payload
            render_log(log_payload)
        else:
            assistant_message["log"] = None

    st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()



