
from __future__ import annotations

import csv
import os

from data import load_dataset
from evidence import resolve_evidence
from fx import FxConverter
from llm import LlmClient, UsageTracker
from plan import decide
from usage_report import write_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def run() -> UsageTracker:
    ds = load_dataset()
    fx = FxConverter(ds)

    usage = UsageTracker()
    llm = LlmClient(usage, ROOT)
    cache_path = os.path.join(ROOT, "code", ".cache", "evidence_cache.json")
    evidence_stats = resolve_evidence(ds, llm, cache_path)

    rows = []
    for request in ds.requests:
        decision = decide(ds, fx, request)
        rows.append(
            [
                request.request_id,
                decision.amount_safe_to_pay,
                decision.affordability_status,
                decision.recommended_payment_method,
                decision.payment_plan,
                decision.earliest_date_for_full_payment,
                decision.spending_changes_needed,
                decision.decision_explanation,
            ]
        )

    out_path = os.path.join(ROOT, "output.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Evidence resolution: {evidence_stats}")
    print(
        f"LLM usage: {len(usage.calls)} calls, "
        f"{usage.total_input_tokens()} input tokens, {usage.total_output_tokens()} output tokens, "
        f"est. cost ${usage.estimated_cost_usd():.4f}"
    )

    write_report(
        usage,
        os.path.join(ROOT, "evaluation", "usage_report.md"),
        run_description=f"Full-dataset run producing output.csv for all {len(ds.requests)} requests in dataset/requests.csv.",
        n_requests=len(ds.requests),
    )
    print(f"Wrote {os.path.join(ROOT, 'evaluation', 'usage_report.md')}")

    return usage


if __name__ == "__main__":
    run()
