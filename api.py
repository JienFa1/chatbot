import logging
import warnings

try:
    from langchain_core.exceptions import LangChainDeprecationWarning
except ImportError:  # pragma: no cover
    class LangChainDeprecationWarning(DeprecationWarning):
        pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'", category=UserWarning, module="sklearn.feature_extraction.text")
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent with your preprocessing. Tokenizing the stop words generated tokens", category=UserWarning, module="sklearn.feature_extraction.text")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from chatbot import Chatbot

app = FastAPI(
    title="Chatbot API",
    description="API for answering questions using the Chatbot",
    version="1.0.0"
)

class Question(BaseModel):
    text: str #attribute (public:+  private:-)  

class Answer(BaseModel):
    response: str

# Initialize chatbot
chatbot = Chatbot()

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        # Get response from chatbot
        response = chatbot.get_response(question.text)
        return Answer(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from typing import List
class AnswerWithContexts(BaseModel):
    response: str
    contexts: List[str]

@app.post("/ask_context", response_model=AnswerWithContexts)
async def ask_question_with_context(question: Question):
    try:
        data = chatbot.get_response_with_context(question.text)
        # đảm bảo contexts là list[str]; nếu chatbot trả về Document, bóc page_content:
        def to_text_list(raw):
            if raw is None: return []
            if isinstance(raw, str): return [raw.strip()] if raw.strip() else []
            out = []
            for it in raw if isinstance(raw, list) else []:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
                elif isinstance(it, dict):
                    pc = it.get("page_content")
                    if isinstance(pc, str) and pc.strip():
                        out.append(pc.strip())
            return out

        return AnswerWithContexts(
            response=data.get("response", ""),
            contexts=to_text_list(data.get("contexts") or data.get("source_documents"))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Chatbot API. Use /ask endpoint to get answers to your questions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 