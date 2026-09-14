from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dataset_path(name: str) -> str:
    return os.path.join(_root(), "dataset", name)


def _parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s)


def _parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "":
        return None
    return float(s)


def _parse_list(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Profile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str]
    expense_categories_to_protect: list[str]
    expense_categories_user_is_willing_to_reduce: list[str]
    expense_categories_user_is_willing_to_stop: list[str]
    payment_methods_user_will_consider: list[str]
    max_installment_months: Optional[int]


@dataclass
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]
    currency: str
    event_date: Optional[date]
    settlement_date: Optional[date]
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[float]


@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: Optional[date]
    payment_frequency_days: Optional[int]
    financing_fee: float
    total_payable_amount: float


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: str
    source_type: str
    message_text: str


@dataclass
class ImageRef:
    image_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]


@dataclass
class Dataset:
    profiles: dict[str, Profile] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    events_by_id: dict[str, Event] = field(default_factory=dict)
    events_by_user: dict[str, list[Event]] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)
    requests_by_id: dict[str, Request] = field(default_factory=dict)
    payment_options_by_request: dict[str, list[PaymentOption]] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    messages_by_request: dict[str, list[Message]] = field(default_factory=dict)
    messages_by_user: dict[str, list[Message]] = field(default_factory=dict)
    messages_by_event: dict[str, list[Message]] = field(default_factory=dict)
    images: list[ImageRef] = field(default_factory=list)
    images_by_event: dict[str, ImageRef] = field(default_factory=dict)
    images_by_request: dict[str, list[ImageRef]] = field(default_factory=dict)
    rates_by_date: dict[date, dict[tuple[str, str], float]] = field(default_factory=dict)


def load_dataset() -> Dataset:
    ds = Dataset()

    for row in _read_csv(_dataset_path("financial_profiles.csv")):
        p = Profile(
            user_id=row["user_id"],
            home_currency=row["home_currency"],
            current_available_balance=_parse_float(row["current_available_balance"]),
            minimum_balance_to_keep=_parse_float(row["minimum_balance_to_keep"]),
            financial_priorities=_parse_list(row["financial_priorities"]),
            expense_categories_to_protect=_parse_list(row["expense_categories_to_protect"]),
            expense_categories_user_is_willing_to_reduce=_parse_list(
                row["expense_categories_user_is_willing_to_reduce"]
            ),
            expense_categories_user_is_willing_to_stop=_parse_list(
                row["expense_categories_user_is_willing_to_stop"]
            ),
            payment_methods_user_will_consider=_parse_list(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=(
                int(float(row["max_installment_months"]))
                if row["max_installment_months"].strip()
                else None
            ),
        )
        ds.profiles[p.user_id] = p

    for row in _read_csv(_dataset_path("financial_events.csv")):
        e = Event(
            event_id=row["event_id"],
            user_id=row["user_id"],
            event_type=row["event_type"],
            description=row["description"],
            category=row["category"],
            direction=row["direction"],
            amount=_parse_float(row["amount"]),
            currency=row["currency"],
            event_date=_parse_date(row["event_date"]),
            settlement_date=_parse_date(row["settlement_date"]),
            status=row["status"],
            linked_event_id=row["linked_event_id"].strip() or None,
            flexibility=row["flexibility"],
            minimum_allowed_amount=_parse_float(row["minimum_allowed_amount"]),
        )
        ds.events.append(e)
        ds.events_by_id[e.event_id] = e
        ds.events_by_user.setdefault(e.user_id, []).append(e)

    for row in _read_csv(_dataset_path("requests.csv")):
        r = Request(
            request_id=row["request_id"],
            user_id=row["user_id"],
            request_date=_parse_date(row["request_date"]),
            request_type=row["request_type"],
            requested_amount=_parse_float(row["requested_amount"]),
            desired_completion_date=_parse_date(row["desired_completion_date"]),
            allows_partial_payment=row["allows_partial_payment"].strip().lower() == "true",
            request_text=row["request_text"],
        )
        ds.requests.append(r)
        ds.requests_by_id[r.request_id] = r

    for row in _read_csv(_dataset_path("request_payment_options.csv")):
        po = PaymentOption(
            payment_option_id=row["payment_option_id"],
            request_id=row["request_id"],
            payment_method=row["payment_method"],
            payment_amount=_parse_float(row["payment_amount"]),
            number_of_payments=int(float(row["number_of_payments"])),
            first_payment_date=_parse_date(row["first_payment_date"]),
            payment_frequency_days=(
                int(float(row["payment_frequency_days"]))
                if row["payment_frequency_days"].strip()
                else None
            ),
            financing_fee=_parse_float(row["financing_fee"]) or 0.0,
            total_payable_amount=_parse_float(row["total_payable_amount"]),
        )
        ds.payment_options_by_request.setdefault(po.request_id, []).append(po)

    for row in _read_csv(_dataset_path("messages.csv")):
        m = Message(
            message_id=row["message_id"],
            user_id=row["user_id"],
            request_id=row["request_id"].strip() or None,
            related_event_id=row["related_event_id"].strip() or None,
            sent_at=row["sent_at"],
            source_type=row["source_type"],
            message_text=row["message_text"],
        )
        ds.messages.append(m)
        if m.request_id:
            ds.messages_by_request.setdefault(m.request_id, []).append(m)
        ds.messages_by_user.setdefault(m.user_id, []).append(m)
        if m.related_event_id:
            ds.messages_by_event.setdefault(m.related_event_id, []).append(m)

    for row in _read_csv(_dataset_path("images.csv")):
        img = ImageRef(
            image_id=row["image_id"],
            user_id=row["user_id"],
            request_id=row["request_id"].strip() or None,
            related_event_id=row["related_event_id"].strip() or None,
        )
        ds.images.append(img)
        if img.related_event_id:
            ds.images_by_event[img.related_event_id] = img
        if img.request_id:
            ds.images_by_request.setdefault(img.request_id, []).append(img)

    for row in _read_csv(_dataset_path("exchange_rates.csv")):
        d = _parse_date(row["rate_date"])
        pair = (row["from_currency"], row["to_currency"])
        rate = _parse_float(row["rate"])
        ds.rates_by_date.setdefault(d, {})[pair] = rate

    return ds


def image_path(image_id: str) -> str:
    return os.path.join(_root(), "dataset", "media", "images", f"{image_id}.png")
