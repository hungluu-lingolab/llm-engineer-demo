"""Tools — Module II, Bài 2 + Bài 4 (Agentic Tool Design & Integration).

15 tool "AI Personal Assistant" chia 5 domain (Bài 4, Section 2 — Hierarchical
Grouping), đủ để minh hoạ tool sprawl thật (ngưỡng chính xác chọn tool giảm
mạnh khi > 20 tool — xem tool_selection.py cho retrieval). Bài 2 chỉ có 3 tool
(check_calendar, send_reminder, search_restaurant); Bài 4 mở rộng domain đó
thành 1 trợ lý cá nhân đầy đủ hơn — KHÔNG giữ nguyên 3 tool cũ, vì mục tiêu là
demo grouping/retrieval thật, không phải tương thích ngược với Bài 2 nữa.

Mỗi tool minh hoạ 1-2 nguyên tắc thiết kế (Bài 4, Section 1):
  - Error handling: try/except, trả lỗi dạng text model đọc được (KHÔNG raise).
  - Idempotency: tool có side-effect nhận `idempotency_key` (mặc định tự sinh
    từ tham số) — gọi lại với cùng key không tạo hiệu ứng nhân đôi.
  - Composite tool: book_dinner_plan gộp 2 bước (check lịch + đặt bàn) —
    minh hoạ trade-off atomic/composite ngay trong domain đã có (Section 1).
"""

from __future__ import annotations

import hashlib

from langchain_core.tools import tool

# ── Idempotency store (Section 1) ──────────────────────────────────────────────
# Demo dùng dict in-memory; production sẽ là bảng DB (unique constraint trên key).
_IDEMPOTENCY_STORE: dict[str, dict] = {}


def _idempotency_key(*parts: str, idempotency_key: str = "") -> str:
    if idempotency_key:
        return idempotency_key
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _run_idempotent(key: str, action_name: str, run) -> dict:
    """Chạy `run()` chỉ nếu key chưa từng thấy; lần sau trả lại kết quả cũ."""
    existing = _IDEMPOTENCY_STORE.get(key)
    if existing is not None:
        return {**existing, "status": "already_done", "idempotency_key": key}
    result = {"action": action_name, "status": "created", "idempotency_key": key, **run()}
    _IDEMPOTENCY_STORE[key] = result
    return result


def _parse_date(value: str) -> None:
    """Validate YYYY-MM-DD. Raise ValueError với message rõ ràng nếu sai."""
    from datetime import date

    date.fromisoformat(value)


# ── Domain: calendar ────────────────────────────────────────────────────────────

@tool
def check_calendar(start_date: str, end_date: str = "") -> str:
    """Kiểm tra lịch làm việc trong một khoảng ngày (YYYY-MM-DD).

    Để trống end_date nếu chỉ hỏi 1 ngày. Với câu hỏi kiểu "tuần này", tự tính
    start_date/end_date rồi gọi MỘT LẦN DUY NHẤT cho cả khoảng.
    """
    try:
        _parse_date(start_date)
        end_date = end_date or start_date
        _parse_date(end_date)
    except ValueError:
        return f"Lỗi: '{start_date}'/'{end_date}' không đúng định dạng YYYY-MM-DD."

    return (
        f"Từ {start_date} đến {end_date}: Thứ 2 14h họp team, "
        f"Thứ 4 16h30 gọi khách hàng, Thứ 6 10h review sprint."
    )


@tool
def create_event(title: str, date: str, time: str, idempotency_key: str = "") -> dict:
    """Tạo sự kiện mới trong lịch. Cần con người phê duyệt (side-effect)."""
    try:
        _parse_date(date)
    except ValueError:
        return {"error": f"Lỗi: '{date}' không đúng định dạng YYYY-MM-DD."}

    key = _idempotency_key("create_event", title, date, time, idempotency_key=idempotency_key)
    return _run_idempotent(
        key, "create_event",
        lambda: {"title": title, "date": date, "time": time},
    )


@tool
def cancel_event(title: str, date: str) -> str:
    """Huỷ một sự kiện đã có trong lịch theo tên + ngày. Cần con người phê duyệt."""
    try:
        _parse_date(date)
    except ValueError:
        return f"Lỗi: '{date}' không đúng định dạng YYYY-MM-DD."
    return f"Đã huỷ sự kiện '{title}' ngày {date}."


# ── Domain: dining ──────────────────────────────────────────────────────────────

@tool
def search_restaurant(location: str, cuisine: str = "") -> str:
    """Tìm nhà hàng theo khu vực và loại món ăn."""
    if not location.strip():
        return "Lỗi: cần cho biết khu vực tìm kiếm (vd: 'Quận 1', 'Cầu Giấy')."
    return f"3 nhà hàng {cuisine or 'đa dạng'} gần {location}: Quán A, Quán B, Quán C."


@tool
def book_table(restaurant: str, party_size: int, time: str, idempotency_key: str = "") -> dict:
    """Đặt bàn tại nhà hàng. Cần con người phê duyệt (side-effect, tốn phí giữ chỗ)."""
    if party_size <= 0:
        return {"error": "Lỗi: số người phải lớn hơn 0."}

    key = _idempotency_key("book_table", restaurant, str(party_size), time, idempotency_key=idempotency_key)
    return _run_idempotent(
        key, "book_table",
        lambda: {"restaurant": restaurant, "party_size": party_size, "time": time},
    )


@tool
def order_food(restaurant: str, items: str, idempotency_key: str = "") -> dict:
    """Đặt món giao tận nơi từ nhà hàng. Cần con người phê duyệt (side-effect, tốn tiền)."""
    if not items.strip():
        return {"error": "Lỗi: chưa chọn món nào để đặt."}

    key = _idempotency_key("order_food", restaurant, items, idempotency_key=idempotency_key)
    return _run_idempotent(
        key, "order_food",
        lambda: {"restaurant": restaurant, "items": items},
    )


# ── Composite tool (Section 1: Atomic vs Composite) ─────────────────────────────
# Gộp check_calendar + book_table thành 1 lời gọi — chỉ nên tồn tại vì đây là
# chuỗi hành động NGƯỜI DÙNG THỰC SỰ hay yêu cầu cùng nhau ("đặt lịch ăn tối"),
# không phải mọi cặp tool đều nên gộp (xem docstring trade-off trong Bài 4).

@tool
def book_dinner_plan(date: str, time: str, restaurant: str, party_size: int) -> dict:
    """Composite: kiểm tra lịch trống RỒI đặt bàn nhà hàng trong 1 lần gọi.

    Dùng khi user yêu cầu "đặt lịch ăn tối" — 1 tác vụ ghép sẵn, không cần
    model tự nối check_calendar → book_table qua 2 lượt riêng. Cần phê duyệt
    (side-effect ở bước đặt bàn).
    """
    try:
        _parse_date(date)
    except ValueError:
        return {"error": f"Lỗi: '{date}' không đúng định dạng YYYY-MM-DD."}

    calendar_check = check_calendar.func(date)
    booking = book_table.func(restaurant, party_size, time)
    return {"calendar": calendar_check, "booking": booking}


# ── Domain: productivity ─────────────────────────────────────────────────────────

@tool
def send_reminder(message: str, time: str, idempotency_key: str = "") -> dict:
    """Đặt lời nhắc. Cần con người phê duyệt trước khi gửi (side-effect)."""
    if not message.strip():
        return {"error": "Lỗi: nội dung lời nhắc không được để trống."}

    key = _idempotency_key("send_reminder", message, time, idempotency_key=idempotency_key)
    return _run_idempotent(
        key, "send_reminder",
        lambda: {"message": message, "time": time},
    )


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """Soạn nháp email (KHÔNG tự gửi — chỉ tạo bản nháp để user tự gửi)."""
    if "@" not in to:
        return f"Lỗi: '{to}' không phải địa chỉ email hợp lệ."
    return f"Đã tạo nháp email tới {to}, tiêu đề: '{subject}' ({len(body)} ký tự nội dung)."


@tool
def create_task(title: str, due_date: str = "", priority: str = "medium") -> str:
    """Tạo task mới trong danh sách việc cần làm."""
    if priority not in ("low", "medium", "high"):
        return f"Lỗi: priority phải là 'low', 'medium', hoặc 'high' (nhận '{priority}')."
    due = f", hạn {due_date}" if due_date else ""
    return f"Đã tạo task '{title}' (ưu tiên {priority}{due})."


# ── Domain: shopping ──────────────────────────────────────────────────────────────

@tool
def search_product(query: str, max_price: float = 0.0) -> str:
    """Tìm sản phẩm theo từ khoá, có thể giới hạn giá tối đa (VNĐ)."""
    if not query.strip():
        return "Lỗi: cần từ khoá tìm kiếm sản phẩm."
    price_note = f" (dưới {max_price:,.0f}đ)" if max_price > 0 else ""
    return f"3 kết quả cho '{query}'{price_note}: Sản phẩm A, Sản phẩm B, Sản phẩm C."


@tool
def track_order(order_id: str) -> str:
    """Tra cứu trạng thái đơn hàng theo mã đơn."""
    if not order_id.strip():
        return "Lỗi: cần mã đơn hàng để tra cứu."
    return f"Đơn {order_id}: đang giao, dự kiến đến trong 2 ngày."


@tool
def add_to_cart(product: str, quantity: int, idempotency_key: str = "") -> dict:
    """Thêm sản phẩm vào giỏ hàng. Cần con người phê duyệt (side-effect nhẹ)."""
    if quantity <= 0:
        return {"error": "Lỗi: số lượng phải lớn hơn 0."}

    key = _idempotency_key("add_to_cart", product, str(quantity), idempotency_key=idempotency_key)
    return _run_idempotent(
        key, "add_to_cart",
        lambda: {"product": product, "quantity": quantity},
    )


# ── Domain: wellness ──────────────────────────────────────────────────────────────

@tool
def log_workout(activity: str, duration_minutes: int) -> str:
    """Ghi lại buổi tập thể dục (loại hoạt động + thời lượng phút)."""
    if duration_minutes <= 0:
        return "Lỗi: thời lượng tập phải lớn hơn 0 phút."
    return f"Đã ghi nhận: {activity}, {duration_minutes} phút."


@tool
def check_sleep_score(date: str = "") -> str:
    """Xem điểm chất lượng giấc ngủ của một ngày (mặc định hôm qua nếu để trống)."""
    if date:
        try:
            _parse_date(date)
        except ValueError:
            return f"Lỗi: '{date}' không đúng định dạng YYYY-MM-DD."
    return f"Điểm giấc ngủ {date or 'hôm qua'}: 78/100 (6h45 phút, 2 lần thức giấc)."


@tool
def find_gym(location: str) -> str:
    """Tìm phòng gym gần một khu vực."""
    if not location.strip():
        return "Lỗi: cần cho biết khu vực tìm kiếm."
    return f"2 phòng gym gần {location}: California Fitness, Elite Gym."


# ── Đăng ký toàn bộ — 15 tool, 5 domain (Bài 4, Section 2: Hierarchical Grouping) ──

TOOL_GROUPS: dict[str, list] = {
    "calendar": [check_calendar, create_event, cancel_event],
    "dining": [search_restaurant, book_table, order_food],
    "productivity": [send_reminder, draft_email, create_task],
    "shopping": [search_product, track_order, add_to_cart],
    "wellness": [log_workout, check_sleep_score, find_gym],
}

# book_dinner_plan không thuộc riêng 1 domain (gộp calendar + dining) — luôn có
# sẵn cho retrieval như 1 tool "ngang hàng" các domain khác (xem tool_selection.py).
COMPOSITE_TOOLS: list = [book_dinner_plan]

TOOLS: list = [t for group in TOOL_GROUPS.values() for t in group] + COMPOSITE_TOOLS

# Tool có side-effect thật — cần HITL approval (interrupt_before=["tools"] trong
# graph.py áp dụng cho MỌI tool call; set này chỉ để UI hiển thị rõ "vì sao cần duyệt").
SENSITIVE_TOOLS = {
    "create_event", "cancel_event",
    "book_table", "order_food", "book_dinner_plan",
    "send_reminder",
    "add_to_cart",
}
