from __future__ import annotations


def clamp_score(value: int | float) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    numeric = max(0.0, min(100.0, numeric))
    return int(round(numeric))

