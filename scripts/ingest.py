"""Ingestion pipeline — Buổi 5 (RAG Pipeline). NATIVE, không LangChain.

Chạy MỘT LẦN để nạp tài liệu vào vector store:
    load_documents → chunk_documents → embed_passages → vectorstore.add

Chạy:
    python -m scripts.ingest                 # ingest từ RAG_SOURCE_DIR (mặc định data/legal_docs)
    python -m scripts.ingest ./duong/dan     # ingest từ thư mục khác

Cần OPENAI_API_KEYS (embedding). Với QDRANT_URL=:memory: dữ liệu KHÔNG persist giữa
các lần chạy — muốn giữ, chạy Qdrant server và đặt QDRANT_URL.
"""

from __future__ import annotations

import sys

from app.config import settings
from app.retrieval import vectorstore
from app.retrieval.chunking import chunk_documents
from app.retrieval.embeddings import embed_passages
from app.retrieval.loader import load_documents


def ingest(source_dir: str | None = None, batch_size: int = 64) -> int:
    source_dir = source_dir or settings.rag_source_dir

    print(f"[1/4] Load tài liệu từ {source_dir} ...")
    docs = load_documents(source_dir)
    print(f"      → {len(docs)} tài liệu/trang")

    print(f"[2/4] Chunking (size={settings.rag_chunk_size}, overlap={settings.rag_chunk_overlap}) ...")
    chunks = chunk_documents(
        docs,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        contextual=settings.rag_contextual_chunking,
    )
    print(f"      → {len(chunks)} chunks")
    if not chunks:
        print("      (không có gì để index — thư mục rỗng?)")
        return 0

    print("[3/4] Embedding + [4/4] Index vào Qdrant (theo batch) ...")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        # id ổn định theo source + chunk_index để chạy lại không tạo trùng.
        ids = [f"{c.metadata.get('source','?')}::{start + i}" for i, c in enumerate(batch)]
        metas = [c.metadata for c in batch]
        vectors = embed_passages(texts)
        vectorstore.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    total = vectorstore.count()
    print(f"\nHoàn tất. Collection '{settings.vectorstore_collection}' có {total} chunks.")
    return total


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    ingest(src)
