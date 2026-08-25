"""Multi-Agent Systems — Module II, Bài 6.

4 pattern (Section 1), mỗi file 1 demo chạy được trên CÙNG domain trợ lý cá
nhân (tái dùng tools.py, TOOL_GROUPS từ Bài 4) để dễ so sánh trực tiếp:

  - sequential.py     : Planner → Scheduler → Notifier (thứ tự cố định)
  - hierarchical.py   : Supervisor điều phối Worker theo domain (nền tảng capstone Bài 7)
  - collaborative.py  : Planner ⇄ Critic phản biện lặp vòng
  - swarm.py          : 3 agent domain ngang hàng, tự handoff (Command)

Tách khỏi app/agent_m2/graph.py (agent đơn, Bài 2-5) vì đây là 4 kiến trúc
ĐỘC LẬP minh hoạ trade-off giữa các pattern, không phải nâng cấp thêm cho
agent hiện có.
"""
