"""
Sliding Window Schema
Converts multi-turn conversations into overlapping window strings.
Exact implementation from v3 notebook Cell 2.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _norm_role(role: Optional[str]) -> str:
    role = str(role or "user").lower()
    return "assistant" if role in ("assistant", "gpt", "bot", "model") else "user"


def _add_turn(turns: list, role: Optional[str], content) -> None:
    if content is None:
        return
    text = str(content).strip()
    if len(text) > 5:
        turns.append({"role": _norm_role(role), "content": text})


def normalize_conversation(row: dict) -> List[Dict]:
    """
    Converts any HF dataset row into a list of {"role", "content"} dicts.
    Handles: messages/conversations list, SoftAge P1/R1 format, flat prompt fields.
    Exact implementation from v3 notebook.
    """
    turns: list = []

    conv = (
        row.get("conversations")
        or row.get("conversation")
        or row.get("messages")
    )
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = (
                msg.get("role")
                or msg.get("from")
                or msg.get("speaker")
                or msg.get("author")
            )
            content = (
                msg.get("content")
                or msg.get("value")
                or msg.get("text")
                or msg.get("message")
            )
            _add_turn(turns, role, content)
        if turns:
            return turns

    # SoftAge format: P1/R1 ... P5/R5
    for i in range(1, 6):
        _add_turn(turns, "user",      row.get(f"P{i}"))
        _add_turn(turns, "assistant", row.get(f"R{i}"))
    if turns:
        return turns

    # Flat single-turn
    text = (
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("text")
        or row.get("Goal")
    )
    _add_turn(turns, "user", text)
    return turns


def conversation_to_windows(
    turns: List[Dict],
    window_size: int = 3,
    stride: int = 1,
    max_chars: int = 512,
) -> List[str]:
    """
    Converts a turn list into overlapping window strings.
    Each window uses role prefixes ([USER]: / [ASST]:) instead of [SEP].
    Truncates from the START to keep the most recent text.
    Returns empty list if no turns.
    """
    if not turns:
        return []

    windows: List[str] = []
    starts = range(0, max(len(turns) - window_size + 1, 1), stride)

    for start in starts:
        chunk = turns[start : start + window_size]
        lines = []
        for t in chunk:
            role   = _norm_role(t.get("role"))
            prefix = "[USER]:" if role == "user" else "[ASST]:"
            lines.append(f"{prefix} {str(t.get('content', '')).strip()}")

        text = "\n".join(lines).strip()

        # Truncate from start — keeps the most recent (most adversarial) content
        if len(text) > max_chars:
            text = text[-max_chars:]

        if len(text) > 20:
            windows.append(text)

    return windows
