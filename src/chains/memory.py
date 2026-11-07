"""Session history: bounded window, query condensation (LLM, with an offline heuristic fallback), reset."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

CUES = {"that", "it", "this", "those", "they", "them", "its", "also", "too", "there", "then"}
STARTS = ("and ", "what about", "how about", "also ", "does that", "is that", "do they", "how long does that")


@dataclass
class SessionMemory:
    """Windowed chat history. Policy: keep the last N turns (a turn = user + assistant message), drop older ones."""
    window_turns: int = 4
    messages: list[tuple[str, str]] = field(default_factory=list)  # (role, content)

    def add(self, role: str, content: str) -> None:
        self.messages.append((role, content))

    def reset(self) -> None:
        self.messages.clear()

    def window(self) -> list[tuple[str, str]]:
        return self.messages[-2 * self.window_turns:]


def to_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    return [HumanMessage(c) if r == "user" else AIMessage(c) for r, c in history]


def history_text(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{'Customer' if r == 'user' else 'Assistant'}: {c}" for r, c in history)


def heuristic_condense(question: str, history: list[tuple[str, str]]) -> str:
    """OFFLINE FALLBACK ONLY (used when no LLM is configured). A short message with a pronoun or an 'and ...' opener is
    treated as a follow-up and joined with the previous customer question; anything else is returned unchanged, so a
    topic switch never inherits stale terms. The LLM condenser handles real ellipsis far better."""
    prev = [c for r, c in history if r == "user"]
    if not prev:
        return question
    q = question.strip()
    words = re.findall(r"[a-z']+", q.lower())
    follow = len(words) <= 7 and (bool(CUES & set(words)) or q.lower().startswith(STARTS))
    return f"{prev[-1].rstrip('?. ')}? {q}" if follow else q
