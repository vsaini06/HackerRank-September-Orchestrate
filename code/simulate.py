
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from data import Dataset, Event
from fx import FxConverter
from recurrence import (
    ENVELOPE_CATEGORIES,
    FIXED_CADENCE_CATEGORIES,
    detect_fixed_recurring_series,
    detect_income_recurring_series,
    detect_spend_envelopes,
    resolve_linked_events,
)

HORIZON_DAYS = 90
EXPLICIT_MATCH_WINDOW_DAYS = 5
EXCLUDED_STATUSES = {"cancelled", "failed", "unrealized"}


@dataclass
class CashFlow:
    on_date: date
    amount_home_ccy: float
    category: str
    source: str


def _dedupe_exact(events: list[Event]) -> list[Event]:
    seen = set()
    out = []
    for e in events:
        key = (e.category, e.description, e.amount, e.currency, e.event_date, e.direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _signed(e: Event) -> float:
    amt = e.amount or 0.0
    return amt if e.direction == "credit" else -amt


def compute_flexible_series(
    ds: Dataset, user_id: str, from_date: date, horizon_days: int = HORIZON_DAYS
):
    horizon_end = from_date + timedelta(days=horizon_days)
    raw_events = ds.events_by_user.get(user_id, [])
    events = resolve_linked_events(_dedupe_exact(raw_events))
    history = [e for e in events if e.event_date and e.event_date <= from_date]
    recurring_series = detect_fixed_recurring_series(history)
    recurring_series += detect_income_recurring_series(events, horizon_end)
    envelopes = detect_spend_envelopes(history, from_date)
    return recurring_series, envelopes


def build_baseline_cashflows(
    ds: Dataset,
    fx: FxConverter,
    user_id: str,
    from_date: date,
    horizon_days: int = HORIZON_DAYS,
    stopped_event_ids: set[str] | None = None,
    reduced_event_amounts: dict[str, float] | None = None,
) -> list[CashFlow]:
    stopped_event_ids = stopped_event_ids or set()
    reduced_event_amounts = reduced_event_amounts or {}

    profile = ds.profiles[user_id]
    horizon_end = from_date + timedelta(days=horizon_days)

    raw_events = ds.events_by_user.get(user_id, [])
    events = resolve_linked_events(_dedupe_exact(raw_events))

    recurring_series, envelopes = compute_flexible_series(ds, user_id, from_date, horizon_days)

    flows: list[CashFlow] = []
    explicit_future_dates_by_category: dict[str, list[date]] = {}
    for e in events:
        d = e.settlement_date or e.event_date
        if d is None or d < from_date or d > horizon_end:
            continue
        if e.status in EXCLUDED_STATUSES:
            continue
        if e.status == "pending" and e.direction == "credit":
            continue
        amount = _signed(e)
        if amount == 0:
            continue
        home_amount = fx.convert(amount, e.currency, profile.home_currency, d)
        flows.append(CashFlow(on_date=d, amount_home_ccy=home_amount, category=e.category, source=e.event_id))
        explicit_future_dates_by_category.setdefault(e.category, []).append(d)


    for series in recurring_series:
        rep_id = series.representative_event.event_id
        if rep_id in stopped_event_ids:
            continue
        amount = reduced_event_amounts.get(rep_id, series.avg_amount)
        signed_amount = amount if series.direction == "credit" else -amount
        explicit_dates = explicit_future_dates_by_category.get(series.category, [])

        next_date = series.last_date
        while True:
            next_date = next_date + timedelta(days=series.cadence_days)
            if next_date > horizon_end:
                break
            if next_date <= from_date:
                continue
            if any(abs((next_date - d).days) <= EXPLICIT_MATCH_WINDOW_DAYS for d in explicit_dates):
                continue
            home_amount = fx.convert(signed_amount, series.currency, profile.home_currency, next_date)
            flows.append(
                CashFlow(on_date=next_date, amount_home_ccy=home_amount, category=series.category, source=f"recurring:{series.category}")
            )

    for env in envelopes:
        rep_id = env.representative_event.event_id
        if rep_id in stopped_event_ids:
            continue
        monthly_amount = env.avg_monthly_amount
        if rep_id in reduced_event_amounts and env.representative_event.amount:
            scale = reduced_event_amounts[rep_id] / env.representative_event.amount
            monthly_amount = env.avg_monthly_amount * max(0.0, scale)

        cycle_date = from_date
        while True:
            cycle_date = cycle_date + timedelta(days=30)
            if cycle_date > horizon_end:
                break
            home_amount = fx.convert(-monthly_amount, env.currency, profile.home_currency, cycle_date)
            flows.append(
                CashFlow(on_date=cycle_date, amount_home_ccy=home_amount, category=env.category, source=f"envelope:{env.category}")
            )

    flows.sort(key=lambda f: f.on_date)
    return flows


@dataclass
class Timeline:
    from_date: date
    horizon_end: date
    starting_balance: float
    checkpoints: list[tuple[date, float]]
    suffix_min_from: dict[date, float]

    def worst_balance_from(self, on_or_after: date) -> float:

        candidates = [b for d, b in self.checkpoints if d >= on_or_after]
        if not candidates:
            return self.starting_balance
        return min(candidates)

    def balance_at(self, on_date: date) -> float:

        best = self.starting_balance
        for d, b in self.checkpoints:
            if d <= on_date:
                best = b
            else:
                break
        return best


def build_timeline(ds: Dataset, fx: FxConverter, user_id: str, from_date: date, flows: list[CashFlow]) -> Timeline:
    profile = ds.profiles[user_id]
    horizon_end = from_date + timedelta(days=HORIZON_DAYS)

    by_date: dict[date, float] = {}
    for f in flows:
        by_date[f.on_date] = by_date.get(f.on_date, 0.0) + f.amount_home_ccy

    running = profile.current_available_balance
    checkpoints: list[tuple[date, float]] = []
    for d in sorted(by_date):
        running += by_date[d]
        checkpoints.append((d, running))

    return Timeline(
        from_date=from_date,
        horizon_end=horizon_end,
        starting_balance=profile.current_available_balance,
        checkpoints=checkpoints,
        suffix_min_from={},
    )


def amount_safe_to_pay(ds: Dataset, timeline: Timeline, user_id: str, requested_amount: float) -> float:

    profile = ds.profiles[user_id]
    worst = min(timeline.starting_balance, timeline.worst_balance_from(timeline.from_date))
    safe = worst - profile.minimum_balance_to_keep
    return max(0.0, min(requested_amount, safe))


def earliest_date_for_full_payment(ds: Dataset, timeline: Timeline, user_id: str, requested_amount: float) -> date | None:

    profile = ds.profiles[user_id]
    minimum = profile.minimum_balance_to_keep

    candidate_dates = [timeline.from_date] + [d for d, _ in timeline.checkpoints]
    for d in sorted(set(candidate_dates)):
        worst_after = timeline.worst_balance_from(d)
        if worst_after - requested_amount >= minimum:
            return d
    return None


def is_plan_safe(ds: Dataset, timeline: Timeline, user_id: str, extra_payments: list[tuple[date, float]]) -> bool:

    profile = ds.profiles[user_id]
    minimum = profile.minimum_balance_to_keep
    if not extra_payments:
        return True

    extra_sorted = sorted(extra_payments, key=lambda p: p[0])
    check_dates = sorted(set([d for d, _ in timeline.checkpoints] + [d for d, _ in extra_sorted] + [timeline.from_date]))

    idx = 0
    cumulative_extra = 0.0
    for d in check_dates:
        while idx < len(extra_sorted) and extra_sorted[idx][0] <= d:
            cumulative_extra += extra_sorted[idx][1]
            idx += 1
        if timeline.balance_at(d) - cumulative_extra < minimum - 1e-6:
            return False
    return True


def simulate_with_overrides(
    ds: Dataset,
    fx: FxConverter,
    user_id: str,
    from_date: date,
    stopped_event_ids: set[str] | None = None,
    reduced_event_amounts: dict[str, float] | None = None,
) -> Timeline:

    flows = build_baseline_cashflows(
        ds, fx, user_id, from_date, stopped_event_ids=stopped_event_ids, reduced_event_amounts=reduced_event_amounts
    )
    return build_timeline(ds, fx, user_id, from_date, flows)
