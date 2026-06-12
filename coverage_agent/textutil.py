"""Small text helpers for report rendering."""
from __future__ import annotations


def truncate_middle(text: str, max_len: int) -> str:
    """Shortens text to max_len by replacing the middle with an ellipsis.

    Keeps the start and end visible — useful for long file paths in
    fixed-width report tables.
    """
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    keep = max_len - 3
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + "..." + (text[-tail:] if tail else "")


def pluralize(count: int, singular: str, plural: str = "") -> str:
    """Returns '<count> <singular|plural>' with a sensible default plural."""
    if count == 1:
        return f"1 {singular}"
    word = plural if plural else singular + "s"
    return f"{count} {word}"
