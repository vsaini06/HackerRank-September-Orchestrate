
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from data import Dataset
from fx import FxConverter
from recurrence import RecurringSeries, SpendEnvelope
from simulate import compute_flexible_series, is_plan_safe, simulate_with_overrides

MAX_SPENDING_CHANGES = 3


@dataclass
class FlexibleTarget:
    category: str
    kind: str
    event_id: str
    new_amount: float

    def as_action(self) -> str:
        if self.kind == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt(self.new_amount)}"


def _fmt(amount: float) -> str:
    return f"{round(amount, 2):g}" if amount == int(amount) else f"{round(amount, 2)}"


def list_flexible_targets(ds: Dataset, user_id: str, from_date: date) -> list[FlexibleTarget]:
    profile = ds.profiles[user_id]
    protect = set(profile.expense_categories_to_protect)
    stoppable_cats = set(profile.expense_categories_user_is_willing_to_stop) - protect
    reducible_cats = set(profile.expense_categories_user_is_willing_to_reduce) - protect

    recurring_series, envelopes = compute_flexible_series(ds, user_id, from_date)
    by_category: dict[str, RecurringSeries | SpendEnvelope] = {}
    for s in recurring_series:
        by_category[s.category] = s
    for e in envelopes:
        by_category[e.category] = e

    targets: list[FlexibleTarget] = []
    for category, series in by_category.items():
        rep = series.representative_event
        flexible = rep.flexibility

        if category in stoppable_cats and flexible in ("stoppable", "reducible_or_stoppable"):
            targets.append(FlexibleTarget(category=category, kind="stop", event_id=rep.event_id, new_amount=0.0))

        if (
            category in reducible_cats
            and flexible in ("reducible", "reducible_or_stoppable")
            and rep.minimum_allowed_amount is not None
            and rep.amount
            and rep.minimum_allowed_amount < rep.amount
        ):
            targets.append(
                FlexibleTarget(category=category, kind="reduce", event_id=rep.event_id, new_amount=rep.minimum_allowed_amount)
            )

    return targets


def _apply(targets: list[FlexibleTarget]) -> tuple[set[str], dict[str, float]]:
    stopped = {t.event_id for t in targets if t.kind == "stop"}
    reduced = {t.event_id: t.new_amount for t in targets if t.kind == "reduce"}
    return stopped, reduced


def find_minimal_safe_changes(
    ds: Dataset,
    fx: FxConverter,
    user_id: str,
    from_date: date,
    extra_payments: list[tuple[date, float]],
    candidates: list[FlexibleTarget] | None = None,
) -> list[FlexibleTarget] | None:
    """Greedily search for the smallest set (<=3, distinct event_ids) of
    stop/reduce actions that makes extra_payments pass the safety check.
    Returns None if no safe combination of <=3 changes exists."""
    if candidates is None:
        candidates = list_flexible_targets(ds, user_id, from_date)
    if not candidates:
        return None

    baseline_worst = simulate_with_overrides(ds, fx, user_id, from_date).worst_balance_from(from_date)

    def worst_with(targets: list[FlexibleTarget]) -> float:
        stopped, reduced = _apply(targets)
        tl = simulate_with_overrides(ds, fx, user_id, from_date, stopped_event_ids=stopped, reduced_event_amounts=reduced)
        return tl.worst_balance_from(from_date)

    scored = sorted(candidates, key=lambda t: -(worst_with([t]) - baseline_worst))

    chosen: list[FlexibleTarget] = []
    used_events: set[str] = set()
    for target in scored:
        if len(chosen) >= MAX_SPENDING_CHANGES:
            break
        if target.event_id in used_events:
            continue
        trial = chosen + [target]
        stopped, reduced = _apply(trial)
        tl = simulate_with_overrides(ds, fx, user_id, from_date, stopped_event_ids=stopped, reduced_event_amounts=reduced)
        if is_plan_safe(ds, tl, user_id, extra_payments):
            return trial
        chosen = trial
        used_events.add(target.event_id)

    return None  # exhausted candidates (or the 3-change cap) without reaching safety
