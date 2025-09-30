# document_processor.py
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from docx import Document as DocxDoc
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from langchain_core.documents import Document as LcDoc
import config
import os

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
def process_documents():
    
    # Khởi tạo embeddings và SemanticChunker
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    chunker = SemanticChunker(
        embeddings,
        buffer_size = 2, breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
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
            
            # Chia nhỏ tài liệu thành các semantic chunks bằng SemanticChunker
            chunks = chunker.split_documents(pages)
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