"""Function calling — Bài 1, Section 5.

Định nghĩa tools (JSON schema) + dispatch tới hàm Python thực tế.
Ở Buổi 1 có một tool demo `lookup_article` trả dữ liệu giả lập — đủ để minh hoạ
flow 4 bước. Từ Buổi 5+ tool này có thể được nối vào retrieval thật.

Lưu ý (Section 5): model KHÔNG tự thực thi function. Nó chỉ trả về tool_call JSON;
code của ta đọc args, gọi hàm, rồi trả kết quả về model.
"""

from __future__ import annotations

# Dữ liệu giả lập cho Buổi 1 — thay bằng tra cứu thật ở các buổi sau.
_FAKE_ARTICLES = {
    "46": "Điều 46 — Công ty TNHH hai thành viên trở lên có từ 02 đến 50 thành viên.",
    "111": "Điều 111 — Công ty cổ phần phải có tối thiểu 03 cổ đông, không giới hạn tối đa.",
}


def lookup_article(article_number: str) -> dict:
    """Hàm Python thực tế (Buổi 1: dữ liệu giả lập)."""
    text = _FAKE_ARTICLES.get(
        article_number, "Không tìm thấy điều luật này trong cơ sở dữ liệu demo."
    )
    return {"article": article_number, "content": text}


# Tool definitions theo đúng format OpenAI (Section 5, slide "Định Nghĩa Tools").
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_article",
            "description": "Tra cứu nội dung một điều trong Luật Doanh nghiệp Việt Nam theo số điều.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "string",
                        "description": "Số điều luật, ví dụ: '46', '111'",
                    }
                },
                "required": ["article_number"],
            },
        },
    }
]

# Map tên function -> hàm Python để dispatch khi nhận tool_call.
DISPATCH = {
    "lookup_article": lookup_article,
}
