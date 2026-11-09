"""Level 3a: payments whose memo says they cover other people.

"covering Amara's share of the airport van, $100 total" is not Jake paying his own shares.
We find who the memo names, then let ARITHMETIC decide: is there an invoice where what those
people (with or without the payer) still owe adds up to the payment exactly?
  * exactly one such invoice  -> split the payment across those shares
  * several, one matching the memo's words (karaoke, firewood...) -> take that one, lower
    invoice confidence
  * none or still ambiguous -> no plan; the payment stays held for a human

Built at payment time against what is still owed, so a share that was already part-paid is
handled correctly. A first name that fits two people ("Wale") is tried as each of them; only
the one whose share fits survives. The model (if any) is used solely to suggest who is
covered when plain name-spotting finds no solution; its suggestion still has to add up exactly.
"""

from dataclasses import dataclass, field
from itertools import product

from models import Invoice, Payment

from . import params as P
from .identity import Directory, memo_name_slots
from .normalize import tokens

MAX_COMBINATIONS = 60


@dataclass
class Split:
    owed_by: str
    invoice_id: str
    cents: int


@dataclass
class Plan:
    payment_id: str
    splits: list[Split]
    invoice_confidence: float
    basis: str = ""  # which learned parameter produced invoice_confidence
    how: str = "memo split"
    notes: list[str] = field(default_factory=list)


def _words(text: str) -> set[str]:
    return {t[:5] for t in tokens(text) if len(t) >= 4}  # crude stem: firewood ~ firewoods


def _groups(slots: list[set[str]], payer: str) -> list[frozenset[str]]:
    combos = list(product(*[sorted(s) for s in slots]))
    if len(combos) > MAX_COMBINATIONS:
        return []
    out = []
    for c in combos:
        g = frozenset(c)
        out += [g, g | {payer}]
    return out


def _solutions(pay: Payment, groups, remaining: dict[str, dict[str, int]], inv_by: dict[str, Invoice]):
    found = []
    for inv_id, owed in remaining.items():
        if inv_by[inv_id].date > pay.timestamp.date():
            continue  # the bill must exist already
        for g in groups:
            if g and all(p in owed for p in g) and sum(owed[p] for p in g) == pay.amount_cents:
                found.append((inv_id, g))
    return found


def plan_memo_payment(pay: Payment, payer: str, directory: Directory,
                      remaining: dict[str, dict[str, int]], inv_by: dict[str, Invoice], llm=None) -> Plan | None:
    """remaining[invoice_id][person_id] = cents still owed right now."""
    slots = memo_name_slots(pay.memo, payer, directory)
    if not slots:
        return None
    sols = _solutions(pay, _groups(slots, payer), remaining, inv_by)
    source = "memo names"
    if not sols and llm:
        suggested = llm.covers(pay, payer, directory.display)
        if suggested:
            sols = _solutions(pay, _groups([{p} for p in suggested], payer), remaining, inv_by)
            source = "model suggested"
    if not sols:
        return None

    if len(sols) > 1:  # let the memo's words pick between invoices that all add up
        memo_words = _words(pay.memo)
        scored = [(len(memo_words & _words(inv_by[i].vendor + " " + inv_by[i].description)), i, g) for i, g in sols]
        best = max(s[0] for s in scored)
        top = [s for s in scored if s[0] == best]
        if best == 0 or len(top) != 1:
            return None  # still ambiguous: leave it for a human
        _, inv_id, group = top[0]
        basis = "inv.memo_picked"
    else:
        inv_id, group = sols[0]
        basis = "inv.memo_unique"
    return Plan(
        pay.id,
        [Split(p, inv_id, remaining[inv_id][p]) for p in sorted(group)],
        P.get(basis),
        basis,
        notes=[f"{source} {sorted(set().union(*slots))}; what {sorted(group)} still owe on {inv_id} adds up to the payment"],
    )
