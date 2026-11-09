"""Shared types for the identity levels: "who sent this payment?"

Identity levels only decide WHO. They never touch invoices. Allocation (allocation.py)
runs afterwards, once, over all payments in time order.
"""

import re
from dataclasses import dataclass, field

from models import Event

from .normalize import normalize, tokens


@dataclass
class Identity:
    payment_id: str
    # person_id -> probability this person sent it. Probabilities sum to less than 1;
    # the remainder means "someone not listed / can't tell".
    candidates: dict[str, float] = field(default_factory=dict)
    resolved: str | None = None  # set only when we are sure enough to act on it
    method: str = "none"  # exact | variant | variant+llm | none
    notes: list[str] = field(default_factory=list)  # why, for the reviewer
    llm_model: str = ""  # which model gave llm_vote
    llm_vote: str | None | bool = False  # False = model not asked; None = it said "none"; else a person id

    @property
    def confidence(self) -> float:
        return self.candidates.get(self.resolved, 0.0) if self.resolved else 0.0


@dataclass
class Directory:
    """Everyone who appears in any event. Person ids are shared across events."""

    display: dict[str, str]  # person_id -> full name
    name_tokens: dict[str, list[str]]  # person_id -> normalised name words
    by_full_name: dict[str, set[str]]  # normalised full name -> person ids
    by_first_name: dict[str, set[str]]  # normalised first name -> person ids
    first_name: dict[str, str]  # person_id -> normalised first name

    @classmethod
    def build(cls, events: list[Event]) -> "Directory":
        d = cls({}, {}, {}, {}, {})
        for ev in events:
            for p in ev.people:
                d.display.setdefault(p.id, p.name)
                words = tokens(p.name)
                d.name_tokens[p.id] = sorted(set(d.name_tokens.get(p.id, [])) | set(words))
                d.by_full_name.setdefault(normalize(p.name), set()).add(p.id)
                d.by_first_name.setdefault(words[0], set()).add(p.id)
                d.first_name.setdefault(p.id, words[0])
        return d

    @property
    def ids(self) -> list[str]:
        return sorted(self.display)


_SUBTRACTION = re.compile(r"\b(?:less|minus|except|excluding|without|not including)\b[^,;+&]*", re.I)


def split_subtractions(memo: str) -> tuple[str, str]:
    """("a + b, less the $48 Mina covered") -> ("a + b, ", " less the $48 Mina covered").
    Whoever or whatever follows "less / minus / except..." is being taken OUT of the payment."""
    return _SUBTRACTION.sub(" ", memo), " ".join(m.group(0) for m in _SUBTRACTION.finditer(memo))


def memo_name_slots(memo: str, payer: str, directory: Directory) -> list[set[str]]:
    """One set of candidate people per distinct first name in the memo ("Wale" may be two people).
    Names mentioned only inside a "less ..." clause don't count."""
    slots, seen = [], set()
    for tok in tokens(split_subtractions(memo)[0]):
        people = {p for p in directory.by_first_name.get(tok, ()) if p != payer}
        if people and tok not in seen:
            seen.add(tok)
            slots.append(people)
    return slots


def memo_names_others(memo: str, payer: str, directory: Directory) -> set[str]:
    """People other than the payer whose first name appears in the memo (outside any
    "less ..." clause: someone mentioned only to be subtracted is not being covered)."""
    return {
        pid
        for tok in tokens(split_subtractions(memo)[0])
        for pid in directory.by_first_name.get(tok, ())
        if pid != payer
    }
