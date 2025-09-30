# config.py
# Danh sách các tài liệu cần xử lý
DOCUMENTS = [
    # Thêm file DOCX
    # Thêm các file khác vào đây
]

FAISS_INDEX_PATH = "data/faiss_index"
OLLAMA_MODEL = "llama3.1:8b-instruct-q6_K"  # Changed from llama2 to llama3.1
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
#cấu hình tiền xử lý văn bản tiếng Việt
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
QUERY_GEN_NUM = 3

# System prompt cho chatbot
SYSTEM_PROMPT = """Tôi là một chatbot được thiết kế để trả lời các câu hỏi về [Tên tài liệu]. 
Tôi có thể:
1. Trả lời các câu hỏi 
2. Giải thích các điều khoản và quy định
3. Cung cấp thông tin về cấu trúc và tổ chức 
4. Hỗ trợ tìm kiếm thông tin cụ thể trong tài liệu được cung cấp.
"""