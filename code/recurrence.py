
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean, median, pstdev

from data import Event

FIXED_CADENCE_CATEGORIES = {
    "rent",
    "utilities",
    "education",
    "debt_repayment",
    "insurance",
    "housing",
    "entertainment",
    "family_support",
    "music_subscription",
    "delivery_membership",
    "cloud_storage",
    "streaming",
    "gym",
    "healthcare",
}

ENVELOPE_CATEGORIES = {"groceries", "transport", "dining", "shopping"}

NON_RECURRING_CATEGORIES = {"investment", "work_expense", "windfall"}

LINKED_ID_CATEGORIES = {"investment", "shopping", "utilities", "work_expense"}

MIN_OCCURRENCES_FOR_RECURRENCE = 2
MAX_INTERVAL_STD_DAYS = 10.0
MAX_INCOME_INTERVAL_STD_DAYS = 15.0
DEFAULT_MONTHLY_CADENCE_DAYS = 30


@dataclass
class RecurringSeries:
    category: str
    direction: str
    avg_amount: float
    cadence_days: int
    last_date: date
    currency: str
    representative_event: Event


@dataclass
class SpendEnvelope:
    category: str
    avg_monthly_amount: float
    currency: str
    representative_event: Event


def resolve_linked_events(events: list[Event]) -> list[Event]:

    by_id = {e.event_id: e for e in events}
    dropped: set[str] = set()

    for e in events:
        if e.category not in LINKED_ID_CATEGORIES:
            continue
        if not e.linked_event_id:
            continue
        origin = by_id.get(e.linked_event_id)
        if origin is None:
            continue

        if e.status == "cancelled" or origin.status == "cancelled":
            dropped.add(e.event_id)
            dropped.add(origin.event_id)
            continue

        if e.status == "settled" and origin.status != "settled":
            dropped.add(origin.event_id)
        elif e.status != "settled" and origin.status == "settled":
            dropped.add(e.event_id)
        elif (e.event_date or date.min) >= (origin.event_date or date.min):
            dropped.add(origin.event_id)
        else:
            dropped.add(e.event_id)

    return [e for e in events if e.event_id not in dropped]


def _cadence_days_for(dates: list[date]) -> tuple[float, float]:
    dates = sorted(dates)
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    return mean(gaps), (pstdev(gaps) if len(gaps) > 1 else 0.0)


def detect_fixed_recurring_series(events: list[Event]) -> list[RecurringSeries]:

    by_category: dict[str, list[Event]] = {}
    for e in events:
        if e.category in FIXED_CADENCE_CATEGORIES and e.status == "settled" and e.event_date:
            by_category.setdefault(e.category, []).append(e)

    series: list[RecurringSeries] = []
    for category, evs in by_category.items():
        evs.sort(key=lambda e: e.event_date)
        if len(evs) < MIN_OCCURRENCES_FOR_RECURRENCE:
            continue
        dates = [e.event_date for e in evs]
        avg_gap, std_gap = _cadence_days_for(dates)
        if std_gap > MAX_INTERVAL_STD_DAYS:
            continue
        amounts = [e.amount for e in evs if e.amount is not None]
        if not amounts:
            continue
        latest = evs[-1]
        series.append(
            RecurringSeries(
                category=category,
                direction=latest.direction,
                avg_amount=mean(amounts[-6:]) if len(amounts) >= 3 else mean(amounts),
                cadence_days=round(avg_gap) or DEFAULT_MONTHLY_CADENCE_DAYS,
                last_date=latest.event_date,
                currency=latest.currency,
                representative_event=latest,
            )
        )
    return series


def detect_income_recurring_series(events: list[Event], horizon_end: date) -> list[RecurringSeries]:

    salary_events = [
        e
        for e in events
        if e.category == "salary"
        and e.status in ("settled", "scheduled")
        and e.event_date
        and e.event_date <= horizon_end
    ]
    salary_events.sort(key=lambda e: e.event_date)
    if len(salary_events) < MIN_OCCURRENCES_FOR_RECURRENCE:
        return []

    dates = [e.event_date for e in salary_events]
    avg_gap, std_gap = _cadence_days_for(dates)
    if std_gap > MAX_INCOME_INTERVAL_STD_DAYS:
        return []

    amounts = [e.amount for e in salary_events if e.amount is not None]
    if not amounts:
        return []

    latest = salary_events[-1]
    return [
        RecurringSeries(
            category="salary",
            direction=latest.direction,
            avg_amount=mean(amounts[-3:]),
            cadence_days=round(avg_gap) or DEFAULT_MONTHLY_CADENCE_DAYS,
            last_date=latest.event_date,
            currency=latest.currency,
            representative_event=latest,
        )
    ]


ENVELOPE_RECENT_WINDOW_DAYS = 90
ENVELOPE_BUCKET_DAYS = 30
MIN_RECENT_BUCKETS_WITH_DATA = 2


def detect_spend_envelopes(events: list[Event], from_date: date) -> list[SpendEnvelope]:

    by_category: dict[str, list[Event]] = {}
    for e in events:
        if e.category in ENVELOPE_CATEGORIES and e.status == "settled" and e.event_date and e.amount:
            by_category.setdefault(e.category, []).append(e)

    envelopes: list[SpendEnvelope] = []
    for category, evs in by_category.items():
        evs.sort(key=lambda e: e.event_date)
        window_start = from_date - timedelta(days=ENVELOPE_RECENT_WINDOW_DAYS)
        recent = [e for e in evs if e.event_date >= window_start]

        buckets = [0.0, 0.0, 0.0]
        buckets_with_data = 0
        for e in recent:
            age_days = (from_date - e.event_date).days
            bucket = min(age_days // ENVELOPE_BUCKET_DAYS, 2)
            if buckets[bucket] == 0.0:
                buckets_with_data += 1
            buckets[bucket] += e.amount

        if buckets_with_data >= MIN_RECENT_BUCKETS_WITH_DATA:
            avg_monthly = median(buckets)
        else:
            span_days = (evs[-1].event_date - evs[0].event_date).days
            months = max(span_days / 30.0, 1.0)
            avg_monthly = sum(e.amount for e in evs) / months

        envelopes.append(
            SpendEnvelope(
                category=category,
                avg_monthly_amount=avg_monthly,
                currency=evs[0].currency,
                representative_event=evs[-1],
            )
        )
    return envelopes
