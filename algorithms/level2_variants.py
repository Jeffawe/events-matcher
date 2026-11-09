"""Level 2 (identity): find name variants with a Fellegi-Sunter style scorer.

For each unresolved payment and each person we ask: how much more likely is this evidence
if the person IS the sender than if they are not? Each piece of evidence contributes
log2(m / u) bits, where
    m = how often the evidence appears when the person really is the sender (hand-set)
    u = how often it appears when someone ELSE is the sender
For NAME evidence u is a fixed small coincidence rate per match level; ambiguity between
people ("Priya" fits two) shows up because their probabilities are then split, not because
the evidence is weakened. For CONTEXT evidence u is measured from the directory.

Evidence:
  name     the best of: every word matches exactly (5) / by prefix or initial, or the full
           name sits inside a longer sender string (4) / one-letter typos (3) / fuzzy:
           nickname inside the name, sounds alike (2) / the first name matches and the
           rest doesn't (1) / nothing (0). Only the best level counts, because these
           overlap and adding them would be double counting.
  context  the person owes something dated by then; the amount equals one of their
           shares; the amount fits within what they owe so far.
  llm      optional vote from a small local model (see llm_ollama.py).

Learning: trusted dictionary aliases are used by level 1. Level 2's own conclusions are
PROPOSED to the dictionary for a human to approve; level 2 never reads its own proposals,
so a wrong early guess cannot reinforce itself.
"""

import math
import re
from datetime import date

from models import Invoice, Payment
from obligations import compute_obligations

from .dictionary import Dictionary
from .identity import Directory, Identity
from . import params as P
from .normalize import tokens

# The probabilities (how often each kind of evidence shows up for the true sender vs for
# someone else) live in params.py, where they start as priors and learn from labelled data.
UNKNOWN_PRIOR = 0.15  # share of payments from someone not on the list at all (not learned)
RESOLVE_P = 0.90  # act on an identity only at or above this probability
# A model's vote may CONFIRM a candidate the rules already favoured strongly; it may not choose
# between close candidates. Small models guess when names are tied ("A KIM": Alex or
# Alexandra?), and a guess must not turn into 99%. Policy threshold, same kind as RESOLVE_P.
LLM_MIN_PRIOR = 0.80
MIN_SHOWN = 0.02  # candidates below this aren't reported
USE_PHONETIC = True  # English-sounding spelling rules; turn off for other languages


# --- string helpers ------------------------------------------------------------------


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(la, lb) // 2 - 1
    ma, mb, m = [False] * la, [False] * lb, 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(lb, i + window + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    jaro = (m / la + m / lb + (m - t / 2) / m) / 3
    prefix = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def phonetic(word: str) -> str:
    """Crude sound key: kris and chris both give 'krs'. Only for plain ascii words."""
    w = word.lower().replace("ph", "f").replace("ch", "k").replace("ck", "k")
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k").replace("y", "i").replace("z", "s")
    w = re.sub(r"(.)\1+", r"\1", w)
    return w[:1] + re.sub(r"[aeiouh]", "", w[1:])


def edit_distance(a: str, b: str) -> int:
    """Edits to turn a into b: insert, delete, substitute, or swap two neighbours."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _exact(t: str, n: str) -> bool:
    return t == n


def _prefix(t: str, n: str) -> bool:
    """t is the name or an initial/prefix of it (j, jo, john), or a nickname extending it (johnny)."""
    return t == n or n.startswith(t) or (len(n) >= 3 and t.startswith(n))


def _typo(t: str, n: str) -> bool:
    """Prefix/initial match, or one slip in a word long enough that one slip is unlikely by chance."""
    return _prefix(t, n) or (len(t) >= 5 and len(n) >= 5 and edit_distance(t, n) <= 1)


def _fuzzy(t: str, n: str) -> bool:
    if _prefix(t, n):
        return True
    if len(t) >= 3 and len(n) >= 3 and jaro_winkler(t, n) >= 0.90:
        return True  # typo
    if len(t) >= 3 and t in n:
        return True  # "mara" inside "amara"
    if len(n) >= 4 and n in t:
        return True  # "tunde" inside "babatunde"
    return (
        USE_PHONETIC and t.isascii() and n.isascii() and len(t) >= 3 and len(n) >= 3
        and phonetic(t) == phonetic(n)
    )


def _covers(sender_words: list[str], name_words: list[str], same) -> bool:
    """Every sender word matches a different word of the person's name."""
    used: set[str] = set()
    for t in sender_words:
        for n in name_words:
            if n not in used and same(t, n):
                used.add(n)
                break
        else:
            return False
    return True


def name_level(sender_words: list[str], name_words: list[str], first_name: str) -> int:
    if not sender_words:
        return 0
    if _covers(sender_words, name_words, _exact):
        return 5
    if _covers(sender_words, name_words, _prefix):
        return 4
    # full name present inside a longer sender string ("Jake Morris Jr", "Mr Jake Morris")
    if len(name_words) >= 2 and all(n in sender_words for n in name_words):
        return 4
    if _covers(sender_words, name_words, _typo):
        return 3
    if _covers(sender_words, name_words, _fuzzy):
        return 2
    if any(len(t) > 1 and t == first_name for t in sender_words):
        return 1  # first name matches, the rest doesn't. A shared surname alone proves nothing.
    return 0


# --- scoring -------------------------------------------------------------------------


def _bits(present: bool, m: float, u: float, n_people: int) -> float:
    """Evidence weight in bits. u is clamped so one-in-N and all-but-one never blow up."""
    u = min(max(u, 1 / n_people), 1 - 1 / n_people)
    return math.log2(m / u) if present else math.log2((1 - m) / (1 - u))


def obligation_index(invoices: list[Invoice]) -> dict[str, list[tuple[date, int]]]:
    out: dict[str, list[tuple[date, int]]] = {}
    for inv in invoices:
        for pid, cents in compute_obligations(inv).owed.items():
            if cents > 0:
                out.setdefault(pid, []).append((inv.date, cents))
    return out


def pair_features(pay: Payment, directory: Directory, owed: dict[str, list[tuple[date, int]]]):
    """For every person: how well their name fits the sender (levels) and the context flags."""
    ids = directory.ids
    sender = tokens(pay.sender_name)
    day = pay.timestamp.date()
    levels = {p: name_level(sender, directory.name_tokens[p], directory.first_name[p]) for p in ids}
    up_to_now = {p: [c for d, c in owed.get(p, []) if d <= day] for p in ids}
    has = {
        "owes": {p: bool(up_to_now[p]) for p in ids},
        "share": {p: pay.amount_cents in up_to_now[p] for p in ids},
        "fits": {p: 0 < pay.amount_cents <= sum(up_to_now[p]) for p in ids},
    }
    return levels, has


def memo_names_a_person(pay: Payment, directory: Directory) -> bool:
    return any(t in directory.by_first_name for t in tokens(pay.memo))


def score_payment(
    pay: Payment,
    directory: Directory,
    owed: dict[str, list[tuple[date, int]]],
    blocked: set[str] = frozenset(),
    llm_vote: str | None | bool = False,  # False = no vote; None = "none"; str = person id
    llm_model: str = "",
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    """Return ({person_id: probability}, {person_id: explanation}, {person_id: name level})."""
    ids = directory.ids
    n = len(ids)
    levels, has = pair_features(pay, directory, owed)
    frac = lambda flags: sum(flags) / n  # noqa: E731

    # odds that a given listed person is the sender, against "nobody listed", before any evidence
    prior = math.log2(((1 - UNKNOWN_PRIOR) / n) / UNKNOWN_PRIOR)
    odds: dict[str, float] = {}
    why: dict[str, str] = {}
    for p in ids:
        if p in blocked:
            continue
        lvl = levels[p]
        if lvl:
            name_w = math.log2(P.get(f"name.m.{lvl}") / P.get(f"name.eps.{lvl}"))
        else:  # nothing like it: evidence against
            name_w = math.log2((1 - P.get("name.m.any")) / (1 - P.get("name.eps.any")))
        bits = prior + name_w
        for flag in ("owes", "share", "fits"):
            if flag == "fits" and memo_names_a_person(pay, directory):
                continue  # people who cover others legitimately pay more than they owe
            bits += _bits(has[flag][p], P.get(f"ctx.m.{flag}"), frac(has[flag].values()), n)
        if llm_vote is not False:
            m, u = P.get(f"llm.m@{llm_model}"), P.get(f"llm.u@{llm_model}")
            bits += math.log2(m / u) if llm_vote == p else math.log2((1 - m) / (1 - u))
        odds[p] = 2**bits
        why[p] = (
            f"name L{lvl}, owes={has['owes'][p]}, amount=share:{has['share'][p]}/fits:{has['fits'][p]}"
            + (", llm agrees" if llm_vote == p else "")
        )

    total = 1 + sum(odds.values())
    probs = {p: o / total for p, o in odds.items() if o / total >= MIN_SHOWN}
    return probs, why, levels


def _can_resolve(top: str | None, probs: dict[str, float], levels: dict[str, int]) -> bool:
    """Context and the model may break ties between people whose names fit the sender.
    They may not make someone resolve whose name looks nothing like it (a memo saying
    "for John" does not make John the sender)."""
    return bool(top) and probs[top] >= RESOLVE_P and levels[top] >= 1


def identify_level2(
    payments: list[Payment],
    directory: Directory,
    invoices: list[Invoice],
    dictionary: Dictionary,
    llm=None,
) -> dict[str, Identity]:
    owed = obligation_index(invoices)
    out: dict[str, Identity] = {}
    for pay in payments:
        _, blocked = dictionary.resolve(pay.sender_name)
        probs, why, levels = score_payment(pay, directory, owed, blocked)
        ident = Identity(pay.id, probs)
        top = max(probs, key=probs.get) if probs else None
        if _can_resolve(top, probs, levels):
            ident.resolved, ident.method = top, "variant"
            ident.notes.append(why[top])
        else:
            # Not sure. Ask the small model, if there is one, and re-score with its vote.
            vote = llm.vote(pay, directory.display) if llm else None
            if vote is not None and vote.available:
                before = probs  # the rules' view, before the model weighed in
                probs, why, levels = score_payment(pay, directory, owed, blocked, vote.person_id, llm.model)
                ident = Identity(pay.id, probs, llm_model=llm.model, llm_vote=vote.person_id)
                ident.notes.append(f"llm said {vote.person_id or 'none'}: {vote.reason}")
                top = max(probs, key=probs.get) if probs else None
                if _can_resolve(top, probs, levels):
                    if before.get(top, 0.0) >= LLM_MIN_PRIOR:
                        ident.resolved, ident.method = top, "variant+llm"
                    else:
                        ident.notes.append(
                            f"model's vote cannot settle it: the rules had {top} at only {before.get(top, 0.0):.0%}")
            if ident.resolved is None:
                ident.notes.append("too uncertain to act on; for review")
                ident.notes += [f"{p} {probs[p]:.0%}: {why[p]}" for p in sorted(probs, key=probs.get, reverse=True)[:3]]
        out[pay.id] = ident
    return out
