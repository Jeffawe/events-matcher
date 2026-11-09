"""Level 4: use the memo's words and amounts to choose WHICH of the payer's invoices.

Levels 1-3 settle who paid and for whom. What is left is "which invoice": with several open
shares the default is oldest first, which is a guess. A memo like "cab 92 + cabin 360",
"badge", or "all three dinner shares" says more.

Method (no data-specific words anywhere):
  1. Take the payer's open shares. A share is RELEVANT to the memo if
       - the memo lists an amount that is (nearly) that share's amount, or
       - the memo shares a word with the invoice's vendor / description / event name.
  2. Look for a set of relevant shares that adds up to the payment.
  3. Exactly one best set -> use it. Ties or nothing -> do nothing (fall back to oldest
     first). The arithmetic is the judge: words only decide which sets to try.
  4. Optional small model: if the words find nothing, ask it which invoices the memo means
     (e.g. "hotel" for "Three rooms, three nights"); its answer must still add up exactly.
"""

import re
from dataclasses import dataclass
from itertools import combinations
from math import comb

from models import Payment

from . import params as P
from .identity import split_subtractions
from .normalize import tokens

AMOUNT_TOLERANCE = 10  # cents: people round the amounts they type ("cab 92" for 91.93)
SUM_TOLERANCE = 2  # cents per share in the set: shares are split with rounding
MAX_POOL, MAX_SET = 16, 6  # bound the subset search
MAX_COINCIDENCES = 0.05  # expected chance matches we tolerate when matching an amount to a set of shares

# Confidences come from params.py (inv.items.amounts / words / model / tie) and are learned.
ADVANCE_PENALTY = 0.10  # taken off when a chosen invoice is dated after the payment

_AMOUNT = re.compile(r"(?<![\w.,])\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?![\w])")


def memo_amounts(memo: str) -> list[int]:
    out = []
    for whole, cents in _AMOUNT.findall(memo):
        c = int(whole.replace(",", "")) * 100 + int((cents or "0").ljust(2, "0"))
        if c >= 100:
            out.append(c)
    return out


def _stems(text: str) -> set[str]:
    return {t[:5] for t in tokens(text) if len(t) >= 4 and not t.isdigit()}


def _explains(word: str, text_words: set[str]) -> bool:
    """A memo word is explained by an invoice's words if one is a prefix of the other (cab ~ cabs)."""
    return any(word.startswith(t) or t.startswith(word) for t in text_words if min(len(word), len(t)) >= 3)


@dataclass
class ItemPick:
    rows: list[list]  # the allocation rows chosen, oldest first
    basis: str  # the learned parameter behind the confidence
    note: str
    penalty: float = 0.0  # taken off for chosen invoices dated after the payment

    @property
    def confidence(self) -> float:
        return round(max(P.get(self.basis) - self.penalty, 0.0), 3)


def choose_items(pay: Payment, pool: list[list], text_of: dict[str, str], llm=None, today=None) -> ItemPick | None:
    """pool rows are [invoice date, invoice id, cents still owed, event id]."""
    if not pool or not pay.memo.strip():
        return None
    kept, taken_out = split_subtractions(pay.memo)  # "... less the $48 Mina covered"
    listed = [a for a in memo_amounts(kept) if a != pay.amount_cents]
    excluded = memo_amounts(taken_out)
    words = _stems(kept)

    def named_by_amount(row):
        return any(abs(row[2] - a) <= AMOUNT_TOLERANCE for a in listed)

    def overlap(row):
        return len(words & _stems(text_of.get(row[1], "")))

    memo_words = {t for t in tokens(kept) if len(t) >= 3 and not t.isdigit()}
    text_words = {r[1]: set(tokens(text_of.get(r[1], ""))) for r in pool}

    relevant = [r for r in pool if (named_by_amount(r) or overlap(r))
                and not any(abs(r[2] - a) <= AMOUNT_TOLERANCE for a in excluded)]
    def explained(rows):  # how many distinct memo words the chosen invoices account for
        return sum(any(_explains(w, text_words[r[1]]) for r in rows) for w in memo_words)

    best = _best_set(pay, relevant, named_by_amount, explained)
    if best:
        if today and any(r[0] > today for r in best.rows):  # an invoice that didn't exist yet
            best.penalty = ADVANCE_PENALTY
            best.note += " (includes an invoice dated after the payment)"
        return best
    if llm and len(pool) <= MAX_POOL:
        ids = llm.pick_invoices(pay, [(r[1], text_of.get(r[1], ""), r[2]) for r in pool])
        chosen = [r for r in pool if r[1] in ids]
        if chosen and _sums(chosen, pay.amount_cents):
            return ItemPick(sorted(chosen, key=lambda r: (r[0], r[1])), "inv.items.model",
                            "model chose the invoices; their shares add up to the payment")
    return None


def _sums(rows: list[list], total: int) -> bool:
    return abs(sum(r[2] for r in rows) - total) <= SUM_TOLERANCE * len(rows)


def _best_set(pay, relevant, named_by_amount, explained) -> ItemPick | None:
    relevant = sorted(relevant, key=lambda r: (r[0], r[1]))[:MAX_POOL]
    found = []
    for k in range(1, min(MAX_SET, len(relevant)) + 1):
        for combo in combinations(relevant, k):
            if _sums(list(combo), pay.amount_cents):
                found.append(list(combo))
    if not found:
        return None
    # strongest evidence first: shares named by an amount, then memo words explained. No
    # "simpler is better" tie-break: two different sets that fit equally well are a tie, and a
    # tie is held, not guessed.
    def strength(c):
        return (sum(map(named_by_amount, c)), explained(c))
    top = max(map(strength, found))
    winners = [c for c in found if strength(c) == top]
    all_by_amount = top[0] == len(winners[0])
    if len(winners) == 1:
        return ItemPick(winners[0], "inv.items.amounts" if all_by_amount else "inv.items.words",
                        "memo's amounts/words pick a set of shares that adds up to the payment")
    # several equally good sets: only acceptable if they are the same amounts on different invoices
    shapes = {tuple(sorted(r[2] for r in c)) for c in winners}
    if len(shapes) == 1:
        return ItemPick(winners[0], "inv.items.tie", f"{len(winners)} equally good sets; took the oldest")
    return None


def exact_subset(pay: Payment, pool: list[list]) -> ItemPick | None:
    """No memo needed: if the payment equals, to the cent, the sum of SOME of the payer's open
    shares, and only one set of shares does, that is almost certainly what was paid, PROVIDED
    the amounts have genuine cents (see below) and the search is small enough that a chance
    match is unlikely. Several sets fit -> no answer (oldest first
    takes over). A single share that already equals the amount is left to the exact-match rule."""
    # Only shares with real cents count. Whole-dollar amounts (120.00, 500.00) make sums collide
    # constantly (120 + 180 = 300, 500 + 500 = 1000), so a match on them proves nothing.
    rows = sorted((r for r in pool if r[2] > 0 and r[2] % 100 != 0), key=lambda r: (r[0], r[1]))[:MAX_POOL]
    if pay.amount_cents % 100 == 0 or any(abs(r[2] - pay.amount_cents) <= 1 for r in pool if r[2] > 0):
        return None
    # How surprising is a match? Searching more subsets makes a coincidence more likely: roughly
    # (subsets searched) x (3 cents of slack) / (range of possible sums). Only search set sizes
    # while the expected number of coincidental matches stays small, or a pool of 16 shares
    # "finds" a wrong combination for nearly any amount.
    total = sum(r[2] for r in rows) or 1
    searched, max_k = 0, 1
    for k in range(2, min(MAX_SET, len(rows)) + 1):
        searched += comb(len(rows), k)
        if searched * 3 / total > MAX_COINCIDENCES:
            break
        max_k = k
    found = [list(c) for k in range(2, max_k + 1) for c in combinations(rows, k)
             if abs(sum(r[2] for r in c) - pay.amount_cents) <= 1]
    if len(found) == 1:
        return ItemPick(found[0], "inv.subset", "the payment equals the exact sum of these open shares, and of no other set")
    return None
