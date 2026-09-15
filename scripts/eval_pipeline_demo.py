"""Demo Module III, Bài 2 — Eval Pipeline chạy trên LangSmith.

Khác scripts/eval_demo.py (Module I, Bài 7 — chạy 1 lần, không version dataset,
không gate theo slice): script này minh hoạ TOÀN BỘ luồng "Từ eval thủ công
sang eval pipeline" (Section 4) + Regression Gate (Section 5):

    golden set YAML (version hoá) → sync LangSmith Dataset → evaluate() chạy
    song song → aggregate (tổng + THEO SLICE) → so baseline → exit 0/1

Chạy:
    python -m scripts.eval_pipeline_demo                      # full dataset
    python -m scripts.eval_pipeline_demo --subset 6            # 6 case đầu (nhanh, rẻ — dùng cho PR)
    python -m scripts.eval_pipeline_demo --skip-ingest         # bỏ qua bước ingest lại RAG data

Cần OPENAI_API_KEYS (judge + RAG) và LANGSMITH_API_KEY (MONITORING_ENABLED
không bắt buộc — script tự tạo LangSmith Client riêng cho phần eval, độc lập
với tracing production).
"""

from __future__ import annotations

import argparse
import sys

from app.eval_pipeline.dataset import load_dataset
from app.eval_pipeline.gate import GateConfig, check_gate, summarize
from app.eval_pipeline.runner import run_eval

DATASET_PATH = "data/eval/legal_qa/v1.yaml"

# Section 5: baseline + drop_tolerance — khớp GATES trong bài học. Baseline ở
# đây là ước lượng ban đầu; production nên tính từ vài lần chạy thật (Section 5:
# "Đo variance nền" trước khi chọn tolerance).
GATES = {
    "overall": GateConfig(baseline=4.0, drop_tolerance=0.5),
    "rule_pass_rate": GateConfig(baseline=0.7, drop_tolerance=0.15),
    "slice:injection": GateConfig(baseline=0.9, drop_tolerance=0.3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH, help="Đường dẫn golden set YAML")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Chỉ chạy N case đầu (nhanh/rẻ — dùng cho eval gate trên PR, Section 6)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Bỏ qua ingest lại RAG data (dùng khi Qdrant server đã có dữ liệu, không phải :memory:)",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        print("[ingest] nạp lại RAG data trước khi eval (bỏ qua bằng --skip-ingest)...")
        from scripts.ingest import ingest

        ingest()

    dataset = load_dataset(args.dataset)
    if args.subset:
        # select_subset ưu tiên phủ ĐỦ slice (round-robin), không cắt N case
        # đầu YAML — tránh bỏ sót hoàn toàn slice rủi ro (injection/out_of_scope)
        # nếu chúng nằm cuối file (Section 6: "ưu tiên slice rủi ro").
        dataset = dataset.select_subset(args.subset)

    print(f"\n[eval] chạy {len(dataset.cases)} case từ '{dataset.name}' v{dataset.version}...")
    results = run_eval(dataset, experiment_prefix=f"{dataset.name}-v{dataset.version}")
    rows = list(results)

    print(f"\nXem chi tiết trên LangSmith: {results.url}\n")

    summary = summarize(rows)
    print("─── Tổng hợp ───")
    print(f"n_cases           = {summary.n}")
    print(f"overall (judge)   = {summary.overall}")
    print(f"rule_pass_rate    = {summary.rule_pass_rate}")
    print("by_slice.type:")
    for slice_type, score in sorted(summary.by_slice_type.items()):
        print(f"  {slice_type:15s} = {score}")

    gate_result = check_gate(summary, GATES)

    print("\n─── Eval Gate ───")
    if gate_result.passed:
        print("PASS — mọi metric trong tolerance so với baseline.")
    else:
        print("FAIL — các metric sau vượt tolerance:")
        for failure in gate_result.failures:
            print(f"  ✗ {failure}")

    sys.exit(0 if gate_result.passed else 1)


if __name__ == "__main__":
    main()
