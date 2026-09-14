
from __future__ import annotations

from collections import deque
from datetime import date, timedelta

from data import Dataset


def _build_graph(rates: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = {}
    for (frm, to), rate in rates.items():
        graph.setdefault(frm, {})[to] = rate
        if rate:
            graph.setdefault(to, {})[frm] = 1.0 / rate
    return graph


def _shortest_path_rate(graph: dict[str, dict[str, float]], frm: str, to: str) -> float | None:
    if frm == to:
        return 1.0
    if frm not in graph:
        return None
    visited = {frm}
    queue = deque([(frm, 1.0)])
    while queue:
        node, acc = queue.popleft()
        for nxt, rate in graph.get(node, {}).items():
            if nxt == to:
                return acc * rate
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, acc * rate))
    return None


class FxConverter:
    def __init__(self, ds: Dataset):
        self._rates_by_date = ds.rates_by_date
        self._sorted_dates = sorted(self._rates_by_date.keys())
        self._graph_cache: dict[date, dict[str, dict[str, float]]] = {}

    def _graph_for(self, d: date) -> dict[str, dict[str, float]]:
        if d not in self._graph_cache:
            self._graph_cache[d] = _build_graph(self._rates_by_date.get(d, {}))
        return self._graph_cache[d]

    def _rate_on(self, d: date, frm: str, to: str) -> float | None:
        return _shortest_path_rate(self._graph_for(d), frm, to)

    def _nearest_dates_outward(self, target: date):
        return sorted(self._sorted_dates, key=lambda d: (abs((d - target).days), d))

    def get_rate(self, on_date: date, from_currency: str, to_currency: str) -> float:
        if from_currency == to_currency:
            return 1.0

        rate = self._rate_on(on_date, from_currency, to_currency)
        if rate is not None:
            return rate

        for candidate in self._nearest_dates_outward(on_date):
            if candidate == on_date:
                continue
            rate = self._rate_on(candidate, from_currency, to_currency)
            if rate is not None:
                return rate

        raise ValueError(
            f"No exchange rate path found from {from_currency} to {to_currency} "
            f"(requested date {on_date}, {len(self._sorted_dates)} rate dates available)"
        )

    def convert(self, amount: float, from_currency: str, to_currency: str, on_date: date) -> float:
        return amount * self.get_rate(on_date, from_currency, to_currency)
