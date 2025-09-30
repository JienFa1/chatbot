# chatbot.py
from retriever import Retriever
from ollama_interface import OllamaInterface

class Chatbot:
    def __init__(self):
        print("Khởi động chatbot...")
        self.retriever = Retriever()
        self.ollama = OllamaInterface()
        
    def get_response(self, question):
        # Get relevant chunks from the retriever
        relevant_chunks = self.retriever.get_relevant_chunks(question)
        
        # Prepare context from relevant chunks
        context = "\n".join(relevant_chunks)
        
        # Get response from Ollama
        response = self.ollama.get_response(question, context)
        return response

    def run(self):
        print("Chatbot sẵn sàng! Gõ 'exit' để thoát.")
        while True:
            user_input = input("Bạn: ")
            if user_input.lower() == 'exit':
                break
                
            response = self.get_response(user_input)
            print(f"Chatbot: {response}")