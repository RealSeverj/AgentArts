"""Referral level labels and colors shared by the report and doctor portal."""
from __future__ import annotations


# Each chip uses text and a border in addition to its background color.
REPORT_LEVEL_COLORS = {
    "L4": ("#991b1b", "#fef2f2", "#fca5a5"),
    "L3": ("#9a3412", "#fff7ed", "#fdba74"),
    "L2": ("#854d0e", "#fefce8", "#fde047"),
    "L1": ("#166534", "#f0fdf4", "#86efac"),
    "L0": ("#475569", "#f1f5f9", "#cbd5e1"),
}


def display_report_level(decision: dict | None) -> str | None:
    """Give completed reports a display level, including items awaiting grading."""
    if not decision or "level" not in decision:
        return None
    level = decision["level"]
    if level is None:
        return "L0"
    return level if isinstance(level, str) and level in REPORT_LEVEL_COLORS else None


def report_level_css() -> str:
    return "\n".join(
        f'.level-chip[data-level="{level}"]{{color:{text};background:{background};border-color:{border}}}'
        for level, (text, background, border) in REPORT_LEVEL_COLORS.items()
    )
