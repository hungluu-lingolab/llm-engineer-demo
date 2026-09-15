"""Scorers — Module III, Bài 2, Section 3 (Eval Methods: chọn cái nào).

3 tầng từ rẻ đến đắt (bài học, "xếp tầng từ rẻ đến đắt"):
  - score_rules  : rule-based assertion (~0đ, tức thì) — must_include/exclude.
  - score_judge  : LLM-as-judge theo rubric — TÁI DÙNG app/eval/judge.py
    (Module I, Bài 7) nguyên xi, không viết lại logic chấm điểm hay bias
    mitigation, chỉ bọc lại đúng chữ ký evaluator mà LangSmith `evaluate()`
    yêu cầu: `(run, example) -> dict`.

Mỗi evaluator nhận `run` (kết quả `target` function vừa chạy — xem runner.py)
và `example` (case gốc từ Dataset, gồm cả `metadata.must_include`/`slice` đã
sync ở dataset.py). Trả về dict `{key, score}` hoặc list các dict đó — đúng
format `EvaluationResult` mà LangSmith evaluate() hiểu.
"""

from __future__ import annotations

import re

from app.eval.judge import judge_answer


def score_rules(run, example) -> dict:
    """Section 3: Rule-based assertion — must_include/must_not_include + regex
    "Điều \\d+" (case cần trích đúng số điều). Không cần LLM, chạy tức thì.

    Trả nhiều metric cùng lúc qua `EvaluationResults` (dict có key "results"
    chứa list) — shape bắt buộc của LangSmith evaluate() khi 1 evaluator chấm
    > 1 tiêu chí (khác `score_judge` chỉ trả 1 dict cho 1 metric).
    """
    output = (run.outputs or {}).get("answer", "") or ""
    metadata = example.metadata or {}
    must_include = metadata.get("must_include", [])
    must_not_include = metadata.get("must_not_include", [])

    output_lower = output.lower()
    include_checks = {kw: kw.lower() in output_lower for kw in must_include}
    exclude_checks = {kw: kw.lower() not in output_lower for kw in must_not_include}
    all_checks = {**include_checks, **exclude_checks}

    passed = all(all_checks.values()) if all_checks else None
    results = [
        {
            "key": "rule_pass",
            "score": passed,
            "comment": _rule_failure_comment(include_checks, exclude_checks),
        }
    ]

    # Case lookup thường kỳ vọng trích đúng số điều — kiểm tra output CÓ chứa
    # ít nhất 1 "Điều N" nếu expected cũng có (không phạt case out_of_scope/injection
    # vốn KHÔNG kỳ vọng trích điều luật).
    expected = (example.outputs or {}).get("expected", "") or ""
    if re.search(r"Điều\s+\d+", expected):
        results.append(
            {"key": "cites_article", "score": bool(re.search(r"Điều\s+\d+", output))}
        )

    return {"results": results}


def _rule_failure_comment(include_checks: dict, exclude_checks: dict) -> str:
    failed_include = [kw for kw, ok in include_checks.items() if not ok]
    failed_exclude = [kw for kw, ok in exclude_checks.items() if not ok]
    parts = []
    if failed_include:
        parts.append(f"thiếu: {failed_include}")
    if failed_exclude:
        parts.append(f"không nên có nhưng có: {failed_exclude}")
    return "; ".join(parts) if parts else "OK"


def score_judge(run, example) -> dict:
    """Section 3: LLM-as-judge theo rubric — bọc lại judge_answer() (Module I,
    Bài 7) đúng chữ ký evaluator của LangSmith. Rubric tuyệt đối + pin model
    version + giảm 6 loại bias đã implement sẵn trong judge.py, KHÔNG lặp lại
    logic ở đây (xem docstring judge.py cho chi tiết bias mitigation).
    """
    output = (run.outputs or {}).get("answer", "") or ""
    question = (example.inputs or {}).get("question", "")
    expected = (example.outputs or {}).get("expected", "") or None

    result = judge_answer(question, output, reference=expected)
    return {
        "key": "judge_overall",
        "score": result.overall,
        "comment": result.reasoning,
        "metadata": {
            "accuracy": result.accuracy,
            "completeness": result.completeness,
            "clarity": result.clarity,
            "groundedness": result.groundedness,
        },
    }


DEFAULT_EVALUATORS = [score_rules, score_judge]
