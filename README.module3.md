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

---

## Cấu trúc (Module III)

```
app/
├── eval_pipeline/         # ✓ Module III — golden dataset + eval runner + regression gate
│   ├── dataset.py         #   load_dataset (YAML), sync_dataset_to_langsmith, select_subset
│   ├── scorers.py         #   score_rules, score_judge (bọc app/eval/judge.py)
│   ├── runner.py          #   run_eval — wraps langsmith.evaluate() + pipeline.answer()
│   └── gate.py            #   summarize (tổng + by_slice), check_gate, diff_failed_cases
data/
└── eval/legal_qa/v1.yaml  # ✓ golden set 30 case (18 lookup, 6 comparison, 3 out_of_scope, 3 injection)
scripts/
└── eval_pipeline_demo.py  # ✓ CLI: sync dataset → evaluate() → report → gate → exit 0/1
.github/workflows/
└── eval-gate.yml          # ✓ CI: subset trên PR (chặn merge), full nightly (không chặn)
```

> Module I/II tái dùng LangSmith **cùng project** cho tracing (`app/monitoring/tracing.py`)
> — golden dataset + eval run của Module III xuất hiện trên cùng dashboard, dùng
> chung 1 `LANGSMITH_API_KEY`.
