
from __future__ import annotations

import json
import os
from datetime import date

from data import Dataset, image_path
from llm import LlmClient

VALID_EFFECTS = {"confirm", "amend_amount", "amend_date", "cancel", "no_new_info"}


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def resolve_evidence(ds: Dataset, llm: LlmClient, cache_path: str) -> dict:
    cache = _load_cache(cache_path)
    stats = {"images_resolved": 0, "messages_resolved": 0, "messages_applied": 0}

    for img in ds.images:
        if not img.related_event_id:
            continue
        event = ds.events_by_id.get(img.related_event_id)
        if event is None or event.amount is not None:
            continue
        key = f"image:{img.image_id}"
        if key not in cache:
            ctx = {"category": event.category, "description": event.description}
            cache[key] = llm.extract_image_amount(image_path(img.image_id), ctx)
        result = cache[key]
        if result.get("amount") is not None:
            event.amount = float(result["amount"])
            stats["images_resolved"] += 1

    for msg in ds.messages:
        if not msg.related_event_id:
            continue
        event = ds.events_by_id.get(msg.related_event_id)
        if event is None:
            continue
        key = f"message:{msg.message_id}"
        if key not in cache:
            ctx = {
                "category": event.category,
                "description": event.description,
                "amount": event.amount,
                "currency": event.currency,
                "event_date": event.event_date.isoformat() if event.event_date else None,
                "settlement_date": event.settlement_date.isoformat() if event.settlement_date else None,
                "status": event.status,
            }
            cache[key] = llm.extract_message_event_effect(msg.message_text, ctx)
        result = cache[key]
        stats["messages_resolved"] += 1

        effect = result.get("effect")
        if effect not in VALID_EFFECTS:
            continue
        if effect == "amend_amount" and result.get("new_amount") is not None:
            event.amount = float(result["new_amount"])
            stats["messages_applied"] += 1
        elif effect == "amend_date" and result.get("new_date"):
            try:
                d = date.fromisoformat(result["new_date"])
                event.event_date = d
                event.settlement_date = d
                stats["messages_applied"] += 1
            except ValueError:
                pass
        elif effect == "cancel":
            event.status = "cancelled"
            stats["messages_applied"] += 1

    _save_cache(cache_path, cache)
    return stats
