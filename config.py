# config.py
# Danh sách các tài liệu cần xử lý
DOCUMENTS = [
    r"C:\Users\TRUYENTHONG\Desktop\chatbot\data\gt-lich-su-dang-csvn-ban-tuyen-giao-tw.pdf",  # Thêm file PDF
    # Thêm file 
]

FAISS_INDEX_PATH = "data/faiss_index"
OLLAMA_MODEL = "llama3.1:8b-instruct-q6_K"  # Changed from llama2 to llama3.1 
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

VI_STOPWORDS = {
    "và","là","các","những","một","của","cho","đến","từ","trong","ngoài","khi","nếu","thì",
    "đã","đang","sẽ","rất","hơn","ít","nhiều","với","về","được","bị","bởi","hay","hoặc",
    "cũng","như","này","kia","đó","ấy","đây","đi","chưa","đừng","không","chỉ","mà","nhưng",
    "vẫn","vừa","cùng","theo","tại","do","vì","qua","lại","sau","trước","giữa","trên","dưới",
    "ra","vào","nên","gồm","mọi","mỗi","tất","cả","vài","nào","đâu","sao","vậy","thế","nơi",
    "hầu","nhằm"
}
VI_STOP_PHRASES = {"vì vậy", "do đó", "tuy nhiên", "mặt khác", "bên cạnh đó", "ngoài ra"}

# Model dùng để sinh subquery (có thể trùng OLLAMA_MODEL)
QUERY_GEN_MODEL = OLLAMA_MODEL
# Số subquery LLM cần tạo (không tính query gốc)
QUERY_GEN_NUM = 2

# System prompt cho chatbot
SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh, giúp trả lời các câu hỏi dựa trên ngữ cảnh được cung cấp.
Yêu cầu:
- Đọc kỹ ngữ cảnh và trả lời chính xác câu hỏi theo ngữ cảnh.
- Nếu ngữ cảnh không đủ để trả lời, hãy thẳng thắn nói rằng bạn không biết, đừng đoán mò.
- Không suy diễn bất kỳ thông tin nào ngoài ngữ cảnh.
"""











VI_STOPWORDS = {
    "và","là","các","những","một","của","cho","đến","từ","trong","ngoài","khi","nếu","thì",
    "đã","đang","sẽ","rất","hơn","ít","nhiều","với","về","được","bị","bởi","hay","hoặc",
    "cũng","như","này","kia","đó","ấy","đây","đi","chưa","đừng","không","chỉ","mà","nhưng",
    "vẫn","vừa","cùng","theo","tại","do","vì","qua","lại","sau","trước","giữa","trên","dưới",
    "ra","vào","nên","gồm","mọi","mỗi","tất","cả","vài","nào","đâu","sao","vậy","thế","nơi",
    "hầu","nhằm"
}
VI_STOP_PHRASES = {"vì vậy", "do đó", "tuy nhiên", "mặt khác", "bên cạnh đó", "ngoài ra"}