# document_processor.py
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from docx import Document as DocxDoc
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from langchain_core.documents import Document as LcDoc
import config
import torch, os, re, logging
from collections import Counter
import unicodedata
from typing import List

class DocxLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def _iter_block_items(self, doc: DocxDoc):
        """Duyệt các block theo đúng thứ tự xuất hiện: Paragraph hoặc Table."""
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield DocxParagraph(child, doc)
            elif isinstance(child, CT_Tbl):
                yield DocxTable(child, doc)

    def load(self):
        doc = DocxDoc(self.file_path)
        lines = []
        table_idx = 0

        for block in self._iter_block_items(doc):
            if isinstance(block, DocxParagraph):
                text = block.text.strip()
                if text:
                    lines.append(text)
            else:
                # block là bảng
                table_idx += 1
                # Nếu muốn đánh dấu bảng, bỏ comment dòng dưới
                # lines.append(f"[TABLE {table_idx}]")
                for row in block.rows:
                    # Gộp text trong từng ô (nhiều paragraph) → một ô
                    cells_text = []
                    for cell in row.cells:
                        cell_text = " ".join(
                            p.text.strip() for p in cell.paragraphs if p.text.strip()
                        )
                        cells_text.append(cell_text)
                    # Chỉ thêm dòng nếu hàng có nội dung
                    if any(t.strip() for t in cells_text):
                        lines.append(" | ".join(cells_text))
                lines.append("")  # dòng trống để tách bảng khỏi nội dung sau

        content = "\n".join(lines).strip()
        return [LcDoc(page_content=content, metadata={"source": self.file_path})]

        # text_lines = [p.text for p in doc.paragraphs if p.text.strip()]
        # content = "\n".join(text_lines)
        # # <- Trả về LangChain Document
        # return [LcDoc(page_content=content, metadata={"source": self.file_path})]
def _clean_text(s: str) -> str:
    if not s:
        return ""
    # loại ký tự điều khiển phổ biến & chuẩn hóa khoảng trắng
    s = (s.replace("\x00", " ")
           .replace("\u200b", " ")   # zero-width space
           .replace("\ufeff", " ")   # BOM
           .replace("\xa0", " "))    # NBSP từ DOCX/PDF
    # gom nhiều khoảng trắng về 1, chuẩn hóa xuống dòng
    s = "\n".join(" ".join(line.split()) for line in s.splitlines())
    return s.strip()

def _strip_boilerplate(text: str) -> str:
    """
    Clean cơ bản: xóa dòng chỉ toàn số (thường là số trang/mục rời),
    chuẩn hóa Unicode NFC, và dọn khoảng trắng/thừa dòng.
    """
    text = text or ""
    # xóa dòng chỉ có số (ví dụ: "12", "3", "218")
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    # chuẩn hóa Unicode (tiếng Việt có dấu)
    text = unicodedata.normalize("NFC", text)
    # bỏ khoảng trắng trước newline, rút gọn nhiều newline liên tiếp
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 2) Remove repeated lines (header/footer lặp) theo nhiều trang
def strip_boilerplate_generic(pages: List[str], min_ratio: float = 0.3) -> List[str]:
    """
    - Áp dụng _strip_boilerplate cho từng trang.
    - Phát hiện dòng lặp (xuất hiện trên >= min_ratio số trang) để loại (header/footer).
    - Thêm một vài pattern 'trung tính' (Page X / Page X of Y, 3/12, v.v.).

    Trả về danh sách trang đã được làm sạch.
    """
    def _basic_clean(t: str) -> str:
        return _strip_boilerplate(t or "")

    pages = [_basic_clean(p) for p in pages]

    if not pages:
        return pages

    # gom các dòng để đếm tần suất xuất hiện toàn corpus
    all_lines, per_page = [], []
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        per_page.append(lines)
        all_lines.extend(lines)

    freq = Counter(all_lines)
    # dòng xuất hiện ≥ min_ratio số trang và không quá dài -> coi là header/footer lặp
    blacklist = {
        l for l, c in freq.items()
        if c / len(pages) >= min_ratio and len(l) <= 120
    }

    # các mẫu "trung tính" thường gặp ở mọi loại tài liệu
    patterns = [
        r"(?i)^page\s+\d+(\s+of\s+\d+)?$",  # Page 3 / Page 3 of 12
        r"(?i)^\d+\s*/\s*\d+\s*$",          # 3/12
    ]
    def looks_like_running_header(line: str) -> bool:
        return any(re.match(p, line) for p in patterns)

    cleaned_pages = []
    for lines in per_page:
        kept = [l for l in lines if l not in blacklist and not looks_like_running_header(l)]
        cleaned_pages.append("\n".join(kept))

    return cleaned_pages


# 3) Lọc chunk rác hiển nhiên (generic, không phụ thuộc domain)
def is_valid_chunk(text: str) -> bool:
    """
    - Loại chuỗi quá ngắn (< 20 ký tự sau strip).
    - Bắt buộc có ít nhất 1 ký tự chữ (A–Z hoặc tiếng Việt có dấu).
    """
    s = (text or "").strip()
    if len(s) < 20:
        return False
    if not re.search(r"[A-Za-zÀ-ỹ]", s):
        return False
    return True

def process_documents():
    
    # Khởi tạo embeddings và SemanticChunker
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL, 
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    chunker = SemanticChunker(
        embeddings,
        buffer_size = 2, breakpoint_threshold_type="standard_deviation", #percentile
        breakpoint_threshold_amount= 0.4, #90
    )
    # RecursiveCharacterTextSplitter(
    #     chunk_size = 800, 
    #     chunk_overlap = 160,
    #     separators = ["\n\n", "\n", " ", ""]
    # ) 
    # Danh sách để lưu tất cả các chunks
    all_chunks = []
    
    # Xử lý từng tài liệu
    for doc_path in config.DOCUMENTS:
        if not os.path.exists(doc_path):
            print(f"Không tìm thấy file: {doc_path}")
            continue
            
        print(f"Đang xử lý file: {doc_path}")
        try:
            # Chọn loader dựa trên phần mở rộng file
            file_extension = os.path.splitext(doc_path)[1].lower()
            if file_extension == '.pdf':
                loader = PyPDFLoader(doc_path)
            elif file_extension == '.txt':
                loader = TextLoader(doc_path, encoding='utf-8')
            elif file_extension == '.docx':
                loader = DocxLoader(doc_path)
            else:
                print(f"Không hỗ trợ định dạng file: {file_extension}")
                continue
            
            # Load tài liệu
            pages = loader.load()

            # 1) Clean cơ bản từng trang
            _pages_raw = []
            for p in pages:
                t = _clean_text(getattr(p, "page_content", ""))
                _pages_raw.append(t)

            # 2) Clean liên-trang (header/footer lặp)
            _pages_clean = strip_boilerplate_generic(_pages_raw, min_ratio=0.25)

            # 3) Gán ngược + loại trang rỗng sau tất cả các bước clean
            clean_pages = []
            for p, t in zip(pages, _pages_clean):
                if t.strip():
                    p.page_content = t.strip()
                    clean_pages.append(p)

            if not clean_pages:
                print("File rỗng sau khi làm sạch, bỏ qua.")
                continue

            # Chia nhỏ tài liệu thành các semantic chunks bằng SemanticChunker
            chunks = chunker.split_documents(clean_pages)
            chunks = [c for c in chunks if is_valid_chunk(c.page_content)]
            all_chunks.extend(chunks)

            
        except Exception as e:
            print(f"Lỗi khi xử lý file {doc_path}: {str(e)}")
            continue
    
    if not all_chunks:
        print("Không có tài liệu nào được xử lý thành công!")
        return
    
    # Tạo vector store từ tất cả các chunks
    print("Đang tạo vector store...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    
    # Lưu vector store
    print(f"Đang lưu vector store vào {config.FAISS_INDEX_PATH}")
    vectorstore.save_local(config.FAISS_INDEX_PATH)
    print("Đã xử lý tài liệu và lưu index FAISS.")