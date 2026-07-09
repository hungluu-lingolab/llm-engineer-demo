"""Demo Buổi 4 — index vài đoạn luật rồi thử semantic search.

Minh hoạ end-to-end: embed_passages → vectorstore.add → embed_query → search.
Đây là "tiền thân" của ingestion pipeline thật ở Buổi 5 (loader + chunking).

Chạy:
    python -m scripts.index_demo

Cần OPENAI_API_KEYS (embedding gọi OpenAI). Qdrant mặc định ":memory:" — mỗi lần
chạy tạo lại từ đầu, không persist (đặt QDRANT_URL để dùng server thật).
"""

from __future__ import annotations

from app.retrieval import vectorstore
from app.retrieval.embeddings import embed_passages, embed_query

# Vài đoạn luật demo (Buổi 5 sẽ thay bằng chunk từ PDF thật).
# Mỗi doc: (id, text, dieu, loai) — 'loai' để minh hoạ metadata filter.
DOCS = [
    ("doc_46", "Điều 46 Luật Doanh nghiệp 2020: Công ty TNHH hai thành viên trở lên có từ 02 đến 50 thành viên.", "46", "tnhh"),
    ("doc_74", "Điều 74: Công ty TNHH một thành viên do một tổ chức hoặc một cá nhân làm chủ sở hữu.", "74", "tnhh"),
    ("doc_111", "Điều 111 Luật Doanh nghiệp 2020: Công ty cổ phần phải có tối thiểu 03 cổ đông, không giới hạn tối đa.", "111", "cophan"),
    ("doc_120", "Điều 120: Cổ phần phổ thông của công ty cổ phần không thể chuyển đổi thành cổ phần ưu đãi.", "120", "cophan"),
    ("doc_47", "Điều 47: Thành viên phải góp đủ vốn cam kết trong 90 ngày kể từ ngày được cấp Giấy chứng nhận đăng ký doanh nghiệp.", "47", "chung"),
]


def index() -> None:
    ids = [d[0] for d in DOCS]
    texts = [d[1] for d in DOCS]
    metas = [{"source": "luat-dn-2020", "dieu": d[2], "loai": d[3]} for d in DOCS]

    print(f"Embedding {len(texts)} đoạn luật...")
    vectors = embed_passages(texts)

    vectorstore.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Đã index. Collection hiện có {vectorstore.count()} documents.\n")


def query(q: str, top_k: int = 3, where: dict | None = None) -> None:
    label = f"Truy vấn: {q!r}"
    if where:
        label += f"   [filter: {where}]"
    print(label)
    hits = vectorstore.search(embed_query(q), top_k=top_k, where=where)
    for h in hits:
        print(f"  [{h.score:.3f}] (Điều {h.metadata.get('dieu')}, loại={h.metadata.get('loai')}) {h.text[:60]}...")
    print()


if __name__ == "__main__":
    index()

    # 1. Semantic search thuần — tìm theo nghĩa, không lọc.
    query("Công ty TNHH tối đa bao nhiêu thành viên?")
    query("Ai làm chủ công ty một thành viên?")

    # 2. Metadata filter — chỉ tìm trong nhóm 'cổ phần'.
    #    Dù câu hỏi hợp với điều luật TNHH hơn, filter loại bỏ chúng khỏi kết quả.
    query("Số lượng thành viên/cổ đông tối thiểu?", where={"loai": "cophan"})

    # 3. Kết hợp nhiều điều kiện filter (source AND loai).
    query("Quy định về vốn góp?", where={"source": "luat-dn-2020", "loai": "chung"})
