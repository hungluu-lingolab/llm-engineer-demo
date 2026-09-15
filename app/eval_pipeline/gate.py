"""Regression Gate — Module III, Bài 2, Section 5 (Regression Testing).

Aggregate kết quả 1 lần chạy (tổng + THEO SLICE — Section 1, "regression vô
hình": điểm tổng ổn nhưng 1 slice tụt hẳn) rồi so với baseline. Gate riêng
cho từng slice quan trọng (vd `slice:injection`) — đúng ví dụ GATES trong bài
học, không chỉ 1 ngưỡng "overall" chung chung.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GateConfig:
    baseline: float
    drop_tolerance: float


@dataclass(slots=True)
class EvalSummary:
    """Aggregate của 1 lần chạy — tổng + theo slice (Section 4: Aggregate)."""

    n: int
    overall: float | None  # trung bình judge_overall (thang 1-5)
    rule_pass_rate: float | None  # tỉ lệ case pass rule_pass (0-1)
    by_slice_type: dict[str, float]  # slice.type -> trung bình judge_overall


def summarize(experiment_results) -> EvalSummary:
    """Section 4: Aggregate — đọc ExperimentResults (langsmith.evaluate()) trực
    tiếp (không qua pandas — to_pandas() kéo thêm dependency không cần thiết
    chỉ để tính vài trung bình)."""
    judge_scores: list[float] = []
    rule_passes: list[bool] = []
    by_slice: dict[str, list[float]] = {}
    n = 0

    for row in experiment_results:
        n += 1
        slice_type = (row["example"].metadata or {}).get("slice", {}).get("type", "unknown")
        for er in row["evaluation_results"]["results"]:
            if er.key == "judge_overall" and er.score is not None:
                judge_scores.append(er.score)
                by_slice.setdefault(slice_type, []).append(er.score)
            elif er.key == "rule_pass" and er.score is not None:
                rule_passes.append(bool(er.score))

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 3) if xs else None

    return EvalSummary(
        n=n,
        overall=mean(judge_scores),
        rule_pass_rate=mean([1.0 if p else 0.0 for p in rule_passes]),
        by_slice_type={k: mean(v) for k, v in by_slice.items() if mean(v) is not None},
    )


@dataclass(slots=True)
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def check_gate(summary: EvalSummary, gates: dict[str, GateConfig]) -> GateResult:
    """Section 5: so summary với baseline theo từng gate — trả fail rõ ràng
    ("metric: giá_trị < baseline - tolerance") để report chỉ đúng chỗ hỏng,
    không chỉ số tổng (bài học: "Report phải chỉ ra case nào rớt")."""
    failures = []
    for metric, cfg in gates.items():
        value = _lookup_metric(summary, metric)
        if value is None:
            continue
        threshold = cfg.baseline - cfg.drop_tolerance
        if value < threshold:
            failures.append(
                f"{metric}: {value:.3f} < {cfg.baseline:.3f} - {cfg.drop_tolerance:.3f} (= {threshold:.3f})"
            )
    return GateResult(passed=len(failures) == 0, failures=failures)


def _lookup_metric(summary: EvalSummary, metric: str) -> float | None:
    if metric == "overall":
        return summary.overall
    if metric == "rule_pass_rate":
        return summary.rule_pass_rate
    if metric.startswith("slice:"):
        slice_type = metric.removeprefix("slice:")
        return summary.by_slice_type.get(slice_type)
    return None


def diff_failed_cases(current_rows, baseline_case_ids_passed: set[str]) -> list[str]:
    """Section 5: liệt kê case CHUYỂN pass -> fail so với 1 baseline run (theo
    case_id) — bài học: số tổng không hành động được, phải xem case-level.

    `baseline_case_ids_passed`: set case_id đã PASS rule_pass ở baseline run
    (caller tự lấy từ 1 EvalSummary/run trước, xem docstring runner.py).
    """
    now_failed = []
    for row in current_rows:
        case_id = (row["example"].metadata or {}).get("case_id")
        rule_pass = next(
            (er.score for er in row["evaluation_results"]["results"] if er.key == "rule_pass"),
            None,
        )
        if case_id in baseline_case_ids_passed and rule_pass is False:
            now_failed.append(case_id)
    return now_failed
