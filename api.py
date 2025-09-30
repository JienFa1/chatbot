from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
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

@app.get("/")
async def root():
    return {"message": "Welcome to Chatbot API. Use /ask endpoint to get answers to your questions."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 