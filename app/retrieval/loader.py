"""Document loader — Buổi 5 (RAG Pipeline), NATIVE (không LangChain).

Load tài liệu từ thư mục, hỗ trợ:
  - .txt / .md : đọc thẳng
  - .pdf       : dùng pypdf (native), mỗi trang là một "page"

Trả về list LoadedDoc (text + metadata source/page). Bước tiếp: chunking.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class LoadedDoc:
    """Một đơn vị tài liệu đã load (một file .txt/.md, hoặc một trang PDF)."""

    text: str
    metadata: dict = field(default_factory=dict)


def _load_text_file(path: Path) -> list[LoadedDoc]:
    text = path.read_text(encoding="utf-8")
    return [LoadedDoc(text=text, metadata={"source": path.name})]


def _load_pdf(path: Path) -> list[LoadedDoc]:
    # Import trong hàm: chỉ cần pypdf khi thực sự có PDF.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[LoadedDoc] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                LoadedDoc(text=text, metadata={"source": path.name, "page": i + 1})
            )
    return docs


# Map đuôi file → hàm load. Thêm .docx/.html sau nếu cần.
_LOADERS = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".pdf": _load_pdf,
}


def load_documents(directory: str) -> list[LoadedDoc]:
    """Load mọi file được hỗ trợ trong `directory` (đệ quy).

    Bỏ qua file không hỗ trợ (không raise) để ingest nhiều loại lẫn lộn không gãy.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")

    docs: list[LoadedDoc] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            docs.extend(_LOADERS[path.suffix.lower()](path))
    return docs
