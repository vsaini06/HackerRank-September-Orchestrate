"""Runs the full pipeline (data loading, FX, evidence resolution, 90-day
simulation, plan ranking) against dataset/sample_requests.csv -- the 25
requests with known-good output columns -- and reports per-field accuracy.

This is a validation tool, not the submission run: it exercises the same
code path as code/main.py but scores it against ground truth instead of
writing output.csv. Its own LLM usage is not what evaluation/usage_report.md
describes; that file summarizes the real code/main.py run over all 250
requests.

Run with:
    python evaluation/main.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from data import Request, load_dataset  # noqa: E402
from evidence import resolve_evidence  # noqa: E402
from fx import FxConverter  # noqa: E402
from llm import LlmClient, UsageTracker  # noqa: E402
from plan import decide  # noqa: E402

FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


def _amount_matches(predicted: float, ground_truth: float, requested_amount: float) -> bool:
    tolerance = max(1.0, 0.05 * requested_amount)
    return abs(predicted - ground_truth) <= tolerance


def main() -> None:
    ds = load_dataset()
    fx = FxConverter(ds)

    usage = UsageTracker()
    llm = LlmClient(usage, ROOT)
    cache_path = os.path.join(ROOT, "code", ".cache", "evidence_cache.json")
    resolve_evidence(ds, llm, cache_path)

    sample_path = os.path.join(ROOT, "dataset", "sample_requests.csv")
    with open(sample_path, encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    correct = {field: 0 for field in FIELDS}
    total = len(samples)
    mismatches: list[str] = []

    for row in samples:
        request = Request(
            request_id=row["request_id"],
            user_id=row["user_id"],
            request_date=date.fromisoformat(row["request_date"]),
            request_type=row["request_type"],
            requested_amount=float(row["requested_amount"]),
            desired_completion_date=date.fromisoformat(row["desired_completion_date"]),
            allows_partial_payment=row["allows_partial_payment"].strip().lower() == "true",
            request_text=row["request_text"],
        )
        decision = decide(ds, fx, request)

        field_ok = {
            "amount_safe_to_pay": _amount_matches(
                decision.amount_safe_to_pay, float(row["amount_safe_to_pay"]), request.requested_amount
            ),
            "affordability_status": decision.affordability_status == row["affordability_status"],
            "recommended_payment_method": decision.recommended_payment_method == row["recommended_payment_method"],
            "payment_plan": decision.payment_plan == row["payment_plan"],
            "earliest_date_for_full_payment": decision.earliest_date_for_full_payment
            == row["earliest_date_for_full_payment"],
            "spending_changes_needed": decision.spending_changes_needed == row["spending_changes_needed"],
        }
        for field, ok in field_ok.items():
            correct[field] += int(ok)

        wrong_fields = [field for field, ok in field_ok.items() if not ok]
        if wrong_fields:
            mismatches.append(f"  {request.request_id}: mismatch on {', '.join(wrong_fields)}")

    print(f"Accuracy over {total} sample requests (dataset/sample_requests.csv):\n")
    for field in FIELDS:
        pct = 100 * correct[field] / total if total else 0.0
        print(f"  {field:35s} {correct[field]:2d}/{total} = {pct:5.1f}%")

    print("\nPer-request mismatches:")
    if mismatches:
        print("\n".join(mismatches))
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
