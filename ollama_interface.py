# ollama_interface.py
import requests
import config

class OllamaInterface:
    def __init__(self):
        self.api_url = config.OLLAMA_API_URL
        self.model = config.OLLAMA_MODEL

    def get_response(self, question, context):
        # Xây dựng prompt với system prompt và context
        prompt = f"""System: {config.SYSTEM_PROMPT}

Context: {context}

Question: {question}

Answer:"""
        
        # Chuẩn bị payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 2048
            }
        }
        
        try:
            # Gửi request đến Ollama API
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Xử lý và trả về response
            result = response.json()
            return result.get("response", "Không thể tạo câu trả lời.")
            
        except Exception as e:
            return f"Lỗi khi gọi Ollama API: {str(e)}"