# Module III — LLM Ops

Module III chuyển từ "xây agent" (Module II) sang "vận hành production": biến các
kỹ thuật eval thủ công (Module I Bài 7, Module II Bài 5) thành **pipeline tự động**
— golden dataset version hoá, chạy song song, chấm điểm nhiều tầng, so baseline,
và chặn merge trên CI khi có regression.

> Cài đặt / chạy / test chung: xem [README.md](README.md).
> Module I (RAG pháp lý): xem [README.module1.md](README.module1.md).
> Module II (AI Agent): xem [README.module2.md](README.module2.md).

---

## Lộ trình xây dựng

| Buổi | Chủ đề | Lắp vào codebase |
|------|--------|------------------|
| **2** | **LLM Evaluation Pipelines** | `eval_pipeline/` — **golden dataset trên LangSmith, rule + judge scorer, regression gate** |
| **3** | **Cost Optimization & Caching** | `cost/` — **token economics, exact/semantic cache 2 tầng, cascading, budget governance** |

### Buổi 2 — LLM Evaluation Pipelines

RAGAS/LLM-as-judge (Module I, Bài 7) và Task Success/Trajectory eval (Module II,
Bài 5) chấm **1 output** tại 1 thời điểm. Buổi này biến chúng thành pipeline chạy
tự động trên toàn bộ golden set, aggregate theo slice (bắt **"regression vô
hình"** — điểm tổng ổn nhưng 1 nhóm case tụt hẳn), và gate chặn merge.

**Golden Dataset (Section 2)** — [`data/eval/legal_qa/v1.yaml`](data/eval/legal_qa/v1.yaml):
30 case theo đúng nội dung THẬT của [`data/legal_docs/luat-doanh-nghiep-2026-trich.md`](data/legal_docs/luat-doanh-nghiep-2026-trich.md)
(Nghị định tiền lương/thù lao — không phải "Luật Doanh nghiệp" như tên file gợi ý),
chia 4 slice:

| Slice | Số case | Kỳ vọng |
|---|---|---|
| `lookup` | 18 | Tra đúng 1 Điều/khoản cụ thể |
| `comparison` (multihop) | 6 | Tổng hợp ≥ 2 phần nội dung (vd so 2 trường hợp lương tối đa) |
| `out_of_scope` | 3 | Hỏi ngoài tài liệu (vd Luật Doanh nghiệp thật) → kỳ vọng từ chối/không suy diễn |
| `injection` | 3 | Prompt injection → kỳ vọng bỏ qua chỉ dẫn độc hại, không lộ system prompt |

Mỗi case: `id`, `question`, `expected`, `slice` (đủ `type`/`difficulty`/`multihop`/`out_of_scope`),
`must_include`/`must_not_include` (rule-based assertion).

[`app/eval_pipeline/dataset.py`](app/eval_pipeline/dataset.py) — `load_dataset()`
đọc + validate YAML (id trùng, question/expected rỗng, thiếu `slice.type` đều
raise ngay). `sync_dataset_to_langsmith()` đồng bộ **YAML → LangSmith Dataset**
theo DIFF (so `case_id`, chỉ update/create/delete phần thay đổi — KHÔNG xoá-tạo-lại
toàn bộ, giữ lịch sử run gắn với example cũ). `split` trên LangSmith = `slice.type`,
filter subset được ngay trên UI.

```bash
python -c "
from app.eval_pipeline.dataset import load_dataset, sync_dataset_to_langsmith
ds = load_dataset('data/eval/legal_qa/v1.yaml')
sync_dataset_to_langsmith(ds)
"
```

> **Vì sao golden set sống ở 2 nơi (YAML + LangSmith)?** YAML là NGUỒN CHÂN LÝ,
> version-controlled trong git, review qua pull request như code (bài học:
> "mỗi case là một 'định nghĩa đúng'; sai một case = sai thước đo mãi mãi").
> LangSmith là bản đồng bộ để `evaluate()` chạy trên đó và tự lưu lịch sử run.

**Eval Methods (Section 3)** — [`app/eval_pipeline/scorers.py`](app/eval_pipeline/scorers.py),
xếp tầng từ rẻ đến đắt:

- `score_rules` — must_include/must_not_include + regex `Điều \d+` (~0đ, tức thì, không cần LLM).
- `score_judge` — **tái dùng nguyên xi** [`app/eval/judge.py`](app/eval/judge.py)
  (Module I, Bài 7: rubric tuyệt đối, 6 loại bias đã giảm thiểu), chỉ bọc lại đúng
  chữ ký evaluator LangSmith cần: `(run, example) -> dict`.

**Runner (Section 4)** — [`app/eval_pipeline/runner.py`](app/eval_pipeline/runner.py)
dùng thẳng `langsmith.evaluate()` thay vì tự viết `ThreadPoolExecutor` + lưu lịch
sử JSON (như code mẫu trong bài học) — LangSmith đã chạy song song trên toàn
Dataset, gọi evaluators, và lưu mỗi lần chạy thành 1 "experiment" gắn với dataset
version. `target` là `pipeline.answer()` (Module I) bọc lại: case `injection` bị
`GuardrailViolation` chặn được coi là **thành công của guardrail**, không phải lỗi.

**Regression Gate (Section 5)** — [`app/eval_pipeline/gate.py`](app/eval_pipeline/gate.py):
`summarize()` đọc `ExperimentResults` trực tiếp (không cần `pandas`), tính tổng +
**theo từng `slice.type`**. `check_gate()` so với baseline + `drop_tolerance` cho
từng metric — kể cả gate riêng cho 1 slice cụ thể (vd `slice:injection`), đúng ví
dụ `GATES` trong bài học. Report chỉ RÕ metric nào rớt (`overall: 4.0 < 4.8 - 0.1`),
không chỉ số tổng.

```bash
# Chạy full 30 case + ingest lại RAG data + áp gate — exit 1 nếu FAIL (dùng cho CI)
python -m scripts.eval_pipeline_demo

# Subset 6 case cho vòng lặp nhanh khi dev (round-robin qua slice — LUÔN phủ đủ
# injection/out_of_scope dù chúng nằm cuối YAML, không cắt N case đầu)
python -m scripts.eval_pipeline_demo --subset 6

# Đã ingest rồi (Qdrant server, không phải :memory:) — bỏ qua ingest lại
python -m scripts.eval_pipeline_demo --skip-ingest
```

> **Vì sao `--subset` không đơn giản là "N case đầu YAML"?** `data/eval/legal_qa/v1.yaml`
> liệt kê 18 case `lookup` trước rồi mới tới `injection`/`out_of_scope` — cắt cơ
> học sẽ bỏ sót HOÀN TOÀN 2 slice rủi ro nhất. `GoldenDataset.select_subset()`
> round-robin qua từng `slice.type` để bất kỳ N nào (kể cả N nhỏ) vẫn phủ đủ 4 slice.

**CI Integration (Section 6)** — [`.github/workflows/eval-gate.yml`](.github/workflows/eval-gate.yml):
2 tầng đúng bài học — PR chạm `prompts/`/`retrieval/`/`eval_pipeline/`/`data/eval/`
chạy **subset 20 case, chặn merge nếu FAIL**; cron nightly chạy **full dataset,
không chặn** (`continue-on-error: true`), chỉ báo cáo xu hướng.

```bash
# .env: cần cả 2 (embedding/chat + đăng ký trace/dataset)
OPENAI_API_KEYS=sk-...
LANGSMITH_API_KEY=lsv2_pt_...
```

### Buổi 3 — Cost Optimization & Caching

Module I, Bài 8 đã giới thiệu 3 đòn bẩy (prompt caching, semantic caching, model
routing). Buổi này **mở rộng sang cấp hệ thống**: đo cost trước khi tối ưu, cache
2 tầng HOÀN CHỈNH với invalidation đúng, cascading (khác routing), và cost
governance. Package mới `app/cost/` — không viết lại semantic cache/routing đã
có, chỉ nâng cấp + bổ sung phần bài học yêu cầu mà Module I chưa có.

**Token Economics + đo trước khi tối ưu (Section 1-2)** — [`app/cost/tracker.py`](app/cost/tracker.py):
`CostTracker` ghi mỗi request (feature/user/model/cache_hit), trả lời đúng 3 lát
cắt bài học nêu — `by_feature()` (tính năng nào ngốn ngân sách), `by_user()` (ai
lạm dụng), `percentile()` (p50 vs p99). `breakdown_tokens()` ước lượng % token
theo nguồn (system/context/history/question) — minh hoạ trực quan **context
bloat là kẻ giết ngân sách**. `app/llm/completion.py` có thêm `chat_with_usage()`
(trả kèm `usage` thật từ API, khác `chat()` chỉ trả text) để tính tiền chính xác
thay vì ước lượng.

```bash
python -c "
from app.cost.tracker import breakdown_tokens, breakdown_pct
pct = breakdown_pct(breakdown_tokens(system='x'*1600, context='y'*12000, history='z'*800, question='q'*400))
print(pct)  # context chiếm phần lớn — giống ví dụ ~70% trong bài học
"
```

**Caching tầng 1: Exact Response Cache (Section 3)** — [`app/cost/cache_exact.py`](app/cost/cache_exact.py):
`ExactCache.get_or_call()` cache TOÀN BỘ câu trả lời cho input giống hệt. Điểm
quan trọng nhất bài học nhấn mạnh: **`prompt_version` PHẢI nằm trong `cache_key()`**
— bump version tự động tạo "namespace" mới, cache cũ hết hạn tự nhiên (đã verify
bằng test: cùng câu hỏi, version cũ → hit; version mới → miss lại).

```bash
python -c "
from app.cost.cache_exact import ExactCache
cache = ExactCache()
n = {'x': 0}
def call(): n['x'] += 1; return f'answer-{n[\"x\"]}'
a1, hit1 = cache.get_or_call(prompt_name='rag', prompt_version=1, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)
a2, hit2 = cache.get_or_call(prompt_name='rag', prompt_version=1, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)
a3, hit3 = cache.get_or_call(prompt_name='rag', prompt_version=2, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)  # bump version
print(hit1, hit2, hit3)  # False True False — v2 KHÔNG trả answer cũ của v1
"
```

> **Vì sao chưa dùng Redis thật?** `CacheStore` là 1 protocol (`get`/`setex`) —
> `InMemoryStore` (ở đây) implement cho demo, không cần Docker. Module III, Bài 4
> (Containerisation) sẽ thêm `RedisStore` implement ĐÚNG 2 method đó vào
> `docker-compose.yml`, KHÔNG đổi gì trong `ExactCache`/logic cache — minh hoạ giá
> trị của việc tách interface store khỏi logic ngay từ đầu.

**Caching tầng 2: Semantic Cache (Section 4)** — nâng cấp [`app/optimization/caching.py`](app/optimization/caching.py)
(giữ tương thích ngược hoàn toàn với `app/api/routes_chat.py` đang dùng `get()`/`set()`):

- `ttl_seconds` — cache stale khi tài liệu nguồn cập nhật.
- `is_volatile()` — loại câu có **năm/tỷ lệ %** cụ thể khỏi semantic cache (false
  hit nguy hiểm nhất theo bài học: "... năm 2020" vs "... năm 2024" similarity
  cao nhưng câu trả lời khác hẳn).
- `stats()` → `hit_rate` — quyết định semantic cache có đáng dùng (bài học: FAQ
  20-40% mới đáng, câu hỏi đa dạng <5% thì không).

**Model Cascading (Section 5)** — [`app/cost/cascade.py`](app/cost/cascade.py):
khác **routing** (`app/optimization/routing.py`, Module I — quyết định model
TRƯỚC dựa trên query), cascading gọi model rẻ TRƯỚC rồi ESCALATE lên model mạnh
dựa trên OUTPUT (không tự tin → escalate). `escalate_rate()` đo tỉ lệ — > 50%
nghĩa là cascade đang LỖ (trả tiền cả 2 model cho phần lớn traffic).

**Cost Governance (Section 7)** — [`app/cost/budget.py`](app/cost/budget.py):
`check_budget()` (spent + dự phóng cuối tháng theo tốc độ chi tiêu, bắn alert ở
50%/80%/100%), `guard_user_budget()` (hard cap/ngày, raise `BudgetExceeded` →
nối vào `app/main.py` exception handler → HTTP 429, giống cách `GuardrailViolation`
→ HTTP 400 ở Module I).

**Hands-on: Cache 2 tầng cho Vietnamese chatbot (Section 8)** — [`scripts/cost_replay_demo.py`](scripts/cost_replay_demo.py)
chạy tập replay [`data/cost/replay_questions.yaml`](data/cost/replay_questions.yaml)
(46 câu ~43% trùng/gần giống, tái dùng 18 câu `lookup` từ golden set Bài 2 làm
nền) qua 4 cấu hình, đúng 5 bước bài học:

```bash
python -m scripts.cost_replay_demo --config no-cache       # baseline
python -m scripts.cost_replay_demo --config exact           # + tầng 1
python -m scripts.cost_replay_demo --config exact+semantic  # + tầng 2
python -m scripts.cost_replay_demo --config cascade         # mở rộng
python -m scripts.cost_replay_demo --compare                # chạy cả 4, in bảng so sánh
```

> **Vì sao exact cache PHẢI check trước semantic cache?** Thứ tự sai (semantic
> trước) khiến semantic "nuốt" luôn cả case trùng y hệt (similarity=1.0 với
> chính câu đã lưu) — `exact_hit_rate` luôn về 0%, dù exact cache có tồn tại.
> Đây là bug thật bắt được khi verify script này lần đầu — kiến trúc đúng
> (Section 4, mermaid tóm tắt bài học) là **tầng 1 → tầng 2 → gọi LLM**, tầng 1
> rẻ hơn (hash lookup, không cần embed) và chính xác 100%.

---

## Cấu trúc (Module III)

```
app/
├── eval_pipeline/         # ✓ Module III, Bài 2 — golden dataset + eval runner + regression gate
│   ├── dataset.py         #   load_dataset (YAML), sync_dataset_to_langsmith, select_subset
│   ├── scorers.py         #   score_rules, score_judge (bọc app/eval/judge.py)
│   ├── runner.py          #   run_eval — wraps langsmith.evaluate() + pipeline.answer()
│   └── gate.py            #   summarize (tổng + by_slice), check_gate, diff_failed_cases
├── cost/                  # ✓ Module III, Bài 3 — token economics + cache 2 tầng + cascade + budget
│   ├── tracker.py         #   CostTracker (by_feature/by_user/percentile), breakdown_tokens
│   ├── cache_exact.py     #   ExactCache (tầng 1), CacheStore protocol, InMemoryStore
│   ├── cascade.py         #   answer_cascade, escalate_rate
│   └── budget.py          #   check_budget, guard_user_budget, BudgetExceeded (→ HTTP 429)
└── optimization/caching.py # ✓ nâng cấp Bài 3: TTL, is_volatile(), stats() (tầng 2, Module I gốc)
data/
├── eval/legal_qa/v1.yaml       # ✓ Bài 2 — golden set 30 case
└── cost/replay_questions.yaml  # ✓ Bài 3 — 46 câu replay (~43% trùng/gần giống)
scripts/
├── eval_pipeline_demo.py  # ✓ Bài 2 — CLI: sync dataset → evaluate() → report → gate → exit 0/1
└── cost_replay_demo.py    # ✓ Bài 3 — CLI: replay 4 cấu hình cache → bảng so sánh cost
.github/workflows/
└── eval-gate.yml          # ✓ CI: subset trên PR (chặn merge), full nightly (không chặn)
```

> Module I/II tái dùng LangSmith **cùng project** cho tracing (`app/monitoring/tracing.py`)
> — golden dataset + eval run của Module III xuất hiện trên cùng dashboard, dùng
> chung 1 `LANGSMITH_API_KEY`.
