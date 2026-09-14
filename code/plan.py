
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from data import Dataset, PaymentOption, Request
from fx import FxConverter
from simulate import amount_safe_to_pay, earliest_date_for_full_payment, is_plan_safe, simulate_with_overrides
from spending_changes import FlexibleTarget, find_minimal_safe_changes, list_flexible_targets


@dataclass
class Candidate:
    method: str 
    payments: list[tuple[date, float]]
    total_paid: float
    payment_option_id: str | None = None
    spending_changes: list[FlexibleTarget] = field(default_factory=list)

    @property
    def first_payment_date(self) -> date:
        return self.payments[0][0]

    @property
    def last_payment_date(self) -> date:
        return self.payments[-1][0]

    def sort_key(self, deadline: date):
        completes = self.last_payment_date <= deadline
        return (
            0 if completes else 1,
            len(self.spending_changes),
            round(self.total_paid, 2),
            self.first_payment_date,
            len(self.payments),
            self.payment_option_id or "",
        )


@dataclass
class Decision:
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


def _installment_schedule(option: PaymentOption) -> list[tuple[date, float]]:
    freq = option.payment_frequency_days or 30
    return [
        (option.first_payment_date + timedelta(days=freq * i), option.payment_amount)
        for i in range(option.number_of_payments)
    ]


def _fmt_amount(amount: float) -> str:
    rounded = round(amount, 2)
    return f"{rounded:.0f}" if rounded == int(rounded) else f"{rounded:g}"


def _plan_string(payments: list[tuple[date, float]]) -> str:
    return "|".join(f"{d.isoformat()}:{_fmt_amount(a)}" for d, a in payments)


def build_candidates(ds: Dataset, fx: FxConverter, user_id: str, request: Request) -> list[Candidate]:
    profile = ds.profiles[user_id]
    accepted = set(profile.payment_methods_user_will_consider)
    timeline = simulate_with_overrides(ds, fx, user_id, request.request_date)
    safe_now = amount_safe_to_pay(ds, timeline, user_id, request.requested_amount)
    earliest_full = earliest_date_for_full_payment(ds, timeline, user_id, request.requested_amount)
    flexible_targets = list_flexible_targets(ds, user_id, request.request_date)

    candidates: list[Candidate] = []

    if "full_payment" in accepted:
        today_payments = [(request.request_date, request.requested_amount)]
        if is_plan_safe(ds, timeline, user_id, today_payments):
            candidates.append(Candidate("full_payment", today_payments, request.requested_amount))
        else:
            changes = find_minimal_safe_changes(ds, fx, user_id, request.request_date, today_payments, flexible_targets)
            if changes is not None:
                candidates.append(Candidate("full_payment", today_payments, request.requested_amount, spending_changes=changes))

        if earliest_full is not None and earliest_full > request.request_date:
            wait_payments = [(earliest_full, request.requested_amount)]
            candidates.append(Candidate("wait", wait_payments, request.requested_amount))

    if request.allows_partial_payment and "partial_payment" in accepted:
        if 0 < safe_now < request.requested_amount and earliest_full is not None and earliest_full <= request.desired_completion_date:
            payments = [(request.request_date, safe_now), (earliest_full, request.requested_amount - safe_now)]
            candidates.append(Candidate("partial_payment", payments, request.requested_amount))

    if "installments" in accepted and profile.max_installment_months is not None:
        for option in ds.payment_options_by_request.get(request.request_id, []):
            if option.payment_method != "installments":
                continue
            if option.number_of_payments > profile.max_installment_months:
                continue
            payments = _installment_schedule(option)
            if is_plan_safe(ds, timeline, user_id, payments):
                candidates.append(Candidate("installments", payments, option.total_payable_amount, payment_option_id=option.payment_option_id))
            else:
                changes = find_minimal_safe_changes(ds, fx, user_id, request.request_date, payments, flexible_targets)
                if changes is not None:
                    candidates.append(
                        Candidate(
                            "installments",
                            payments,
                            option.total_payable_amount,
                            payment_option_id=option.payment_option_id,
                            spending_changes=changes,
                        )
                    )

    return candidates


def _explanation(status: str, method: str, winner: Candidate | None, request: Request, profile, currency: str) -> str:
    min_bal = _fmt_amount(profile.minimum_balance_to_keep)
    if status == "affordable_now":
        return f"Pay {currency} {_fmt_amount(request.requested_amount)} today. This keeps the {currency} {min_bal} minimum protected over the next 90 days."
    if status == "affordable_with_plan" and method == "partial_payment":
        first, second = winner.payments
        return (
            f"Pay {currency} {_fmt_amount(first[1])} today and the remaining {currency} {_fmt_amount(second[1])} "
            f"on {second[0].isoformat()}. This completes the full request and keeps the {currency} {min_bal} minimum protected."
        )
    if status == "affordable_with_plan" and method == "installments":
        n = len(winner.payments)
        amt = winner.payments[0][1]
        start = winner.first_payment_date.isoformat()
        return f"Use {n} installments of {currency} {_fmt_amount(amt)}, starting {start}. This keeps the {currency} {min_bal} minimum protected."
    if status == "affordable_with_plan" and method == "full_payment":
        changes_txt = "; ".join(_change_text(c) for c in winner.spending_changes)
        return f"{changes_txt}, then pay {currency} {_fmt_amount(request.requested_amount)} today. This keeps the {currency} {min_bal} minimum protected."
    if status == "affordable_later":
        return (
            f"Pay {currency} {_fmt_amount(request.requested_amount)} in full on {winner.first_payment_date.isoformat()}. "
            f"Paying earlier would take the balance below the {currency} {min_bal} minimum."
        )
    return (
        f"Do not proceed with the {currency} {_fmt_amount(request.requested_amount)} request by {request.desired_completion_date.isoformat()}. "
        f"None of the available options keeps the {currency} {min_bal} minimum protected."
    )


def _change_text(c: FlexibleTarget) -> str:
    if c.kind == "stop":
        return f"Stop the {c.category} expense"
    return f"Reduce the {c.category} expense to {_fmt_amount(c.new_amount)}"


def decide(ds: Dataset, fx: FxConverter, request: Request) -> Decision:
    user_id = request.user_id
    profile = ds.profiles[user_id]
    currency = profile.home_currency

    timeline = simulate_with_overrides(ds, fx, user_id, request.request_date)
    safe_now = amount_safe_to_pay(ds, timeline, user_id, request.requested_amount)
    earliest_full = earliest_date_for_full_payment(ds, timeline, user_id, request.requested_amount)

    candidates = build_candidates(ds, fx, user_id, request)

    if not candidates:
        return Decision(
            amount_safe_to_pay=round(safe_now, 2),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=earliest_full.isoformat() if earliest_full else "",
            spending_changes_needed="none",
            decision_explanation=_explanation("not_affordable", "not_recommended", None, request, profile, currency),
        )

    winner = min(candidates, key=lambda c: c.sort_key(request.desired_completion_date))

    if winner.method == "full_payment" and winner.first_payment_date == request.request_date and not winner.spending_changes:
        status = "affordable_now"
    elif winner.method == "wait":
        status = "affordable_later"
    else:
        status = "affordable_with_plan"

    spending_changes_str = "|".join(c.as_action() for c in winner.spending_changes) or "none"

    return Decision(
        amount_safe_to_pay=round(safe_now, 2),
        affordability_status=status,
        recommended_payment_method=winner.method,
        payment_plan=_plan_string(winner.payments),
        earliest_date_for_full_payment=earliest_full.isoformat() if earliest_full else "",
        spending_changes_needed=spending_changes_str,
        decision_explanation=_explanation(status, winner.method, winner, request, profile, currency),
    )
