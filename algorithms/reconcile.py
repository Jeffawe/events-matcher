"""Level 3b: the surplus and deficit view.

After allocation, for every person and event: what they owe, what has been paid toward
it (by anyone), and what is still open. Money that could not be placed (a surplus with a
known payer) is lined up against people who still owe, as RANKED SUGGESTIONS for a human.
Nothing here changes any link.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from models import Event, Invoice, Payment, PersonBalance, UnmatchedAmount
from obligations import compute_obligations

from .allocation import AllocationResult
from .identity import Directory
from .normalize import tokens


ROUNDING_LIMIT_CENTS = 100


@dataclass
class Suggestion:
    payment_id: str
    payer_id: str
    cents: int  # unplaced money
    person_id: str  # who it might be for
    invoice_id: str
    open_cents: int  # what that person still owes on that invoice
    score: int
    why: list[str]


@dataclass
class Reconciliation:
    balances: list[PersonBalance] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    unexplained_cents: int = 0  # surplus with no plausible home
    rounding_cents: int = 0  # leftovers under $1, treated as rounding, not suggested anywhere


def reconcile(
    invoices: list[Invoice],
    payments: list[Payment],
    alloc: AllocationResult,
    directory: Directory,
    events: list[Event],
) -> Reconciliation:
    inv_event = {i.id: i.event_id for i in invoices}
    owed: dict[tuple[str, str], int] = defaultdict(int)
    for inv in invoices:
        for pid, c in compute_obligations(inv).owed.items():
            owed[(pid, inv.event_id)] += c
    paid: dict[tuple[str, str], int] = defaultdict(int)
    for l in alloc.links:
        paid[(l.owed_by, inv_event[l.invoice_id])] += l.amount_cents
    out = Reconciliation([PersonBalance(p, e, c, paid[(p, e)]) for (p, e), c in sorted(owed.items())])

    member = defaultdict(set)  # person -> events they belong to
    for ev in events:
        for p in ev.people:
            member[p.id].add(ev.id)
    memo_of = {p.id: p.memo for p in payments}

    leftovers = [u for u in alloc.unmatched
                 if u.payer_id and (u.reason.startswith("surplus") or u.reason.startswith("nothing open")
                                    or u.reason.startswith("memo names"))]
    for u in leftovers:
        if u.amount_cents < ROUNDING_LIMIT_CENTS:
            out.rounding_cents += u.amount_cents
            continue
        named = {pid for t in tokens(memo_of[u.payment_id]) for pid in directory.by_first_name.get(t, ())}
        scored: list[Suggestion] = []
        for person, open_rows in alloc.remaining.items():
            for inv_id, ev_id, open_c in open_rows:
                why, score = [], 0
                if person in named and person != u.payer_id:
                    why.append("memo names them"); score += 3
                if open_c == u.amount_cents:
                    why.append("amount equals what they still owe on this invoice"); score += 2
                if ev_id in member[u.payer_id]:
                    why.append("same event as the payer"); score += 1
                if person == u.payer_id:
                    why.append("payer's own later invoice (advance?)")
                if score >= 3:
                    scored.append(Suggestion(u.payment_id, u.payer_id, u.amount_cents, person, inv_id, open_c, score, why))
        scored.sort(key=lambda s: -s.score)
        out.suggestions += scored[:3]
        if not scored:
            out.unexplained_cents += u.amount_cents
    return out
