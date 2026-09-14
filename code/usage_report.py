
from __future__ import annotations

import os
from datetime import datetime, timezone

from llm import PRICING_PER_MTOK, UsageTracker


def write_report(usage: UsageTracker, path: str, run_description: str, n_requests: int) -> None:
    by_model: dict[str, dict[str, int]] = {}
    by_purpose: dict[str, dict[str, int]] = {}
    for c in usage.calls:
        m = by_model.setdefault(c.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        m["calls"] += 1
        m["input_tokens"] += c.input_tokens
        m["output_tokens"] += c.output_tokens

        p = by_purpose.setdefault(c.purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        p["calls"] += 1
        p["input_tokens"] += c.input_tokens
        p["output_tokens"] += c.output_tokens

    total_calls = len(usage.calls)
    total_in = usage.total_input_tokens()
    total_out = usage.total_output_tokens()
    total_tokens = total_in + total_out
    total_cost = usage.estimated_cost_usd()

    lines = []
    lines.append("# Token Usage and Cost Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"Run: {run_description}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Model provider: Anthropic")
    lines.append(f"- Total model calls: {total_calls}")
    lines.append(f"- Total input tokens: {total_in}")
    lines.append(f"- Total output tokens: {total_out}")
    lines.append(f"- Total tokens: {total_tokens}")
    lines.append(f"- Requests processed: {n_requests}")
    if n_requests:
        lines.append(f"- Average tokens per request: {total_tokens / n_requests:.2f}")
    lines.append(f"- Estimated total cost (USD): ${total_cost:.4f}")
    if n_requests:
        lines.append(f"- Estimated cost per request (USD): ${total_cost / n_requests:.6f}")
    lines.append("- Pricing basis (approximate, per model card, per MTok):")
    for model in sorted(PRICING_PER_MTOK):
        p = PRICING_PER_MTOK[model]
        lines.append(f"  - {model}: ${p['input']:.2f} input / ${p['output']:.2f} output")
    lines.append("")

    lines.append("## Per-model breakdown")
    lines.append("")
    lines.append("| Model | Calls | Input tokens | Output tokens | Est. cost (USD) |")
    lines.append("|---|---|---|---|---|")
    for model, s in sorted(by_model.items()):
        pricing = PRICING_PER_MTOK.get(model, {"input": 0.0, "output": 0.0})
        cost = s["input_tokens"] / 1_000_000 * pricing["input"] + s["output_tokens"] / 1_000_000 * pricing["output"]
        lines.append(f"| {model} | {s['calls']} | {s['input_tokens']} | {s['output_tokens']} | ${cost:.4f} |")
    lines.append("")

    lines.append("## Per-purpose breakdown")
    lines.append("")
    lines.append("| Purpose | Calls | Input tokens | Output tokens |")
    lines.append("|---|---|---|---|")
    for purpose, s in sorted(by_purpose.items()):
        lines.append(f"| {purpose} | {s['calls']} | {s['input_tokens']} | {s['output_tokens']} |")
    lines.append("")

    if total_calls == 0:
        lines.append(
            "_No model calls were made in this run (evidence cache fully warm, or no "
            "messages/images required interpretation for this dataset slice)._"
        )
        lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
