"""Allocation: put identified payments onto invoice shares.

One pass over ALL payments in time order, so every payment sees what earlier ones already
paid. It is a pure function of its inputs: when an identity changes, run it again.

Rules:
  * Ceiling: never apply more than a share's remaining amount. Money beyond what the
    sender owes is NOT forced onto anything; it comes back as unmatched with the payer
    attached (a surplus).
  * A memo-explained plan (level3_memo) is applied as given: each person's share on the
    invoice the arithmetic found.
  * Otherwise the money goes to the sender's own shares:
      - If the memo names exactly one event ("Austin", "deposit for wedding"), that event's
        shares are used first, even if the invoice is dated later (deposits).
      - Then shares of invoices dated on or before the payment: an amount equal to a
        remaining share is taken as "exact" (n equal shares: oldest, invoice confidence
        1/n), else oldest first. Shares may be part-paid.
  * Held, not allocated: sender not identified; memo names another person and nothing
    adds up; not incoming money ("requests $100", "You paid ..."); bank alert or likely
    duplicate of a payment minutes earlier.

Links carry two confidences: person_confidence (who the money belongs to) and
invoice_confidence (which of that person's invoices). A later level may move a link to
another invoice of the same person when invoice_confidence < 1.
"""

import re
from dataclasses import dataclass, field
from datetime import timedelta

from models import Event, Invoice, Payment, PaymentLink, UnmatchedAmount
from obligations import compute_obligations

from .identity import Directory, Identity, memo_names_others
from .lexicon import BANK_ALERT, NOT_INCOMING, SAYS_SEPARATE
from . import params as P
from .level3_memo import Plan, plan_memo_payment
from .level4_items import SUM_TOLERANCE, choose_items, exact_subset
from .normalize import normalize, tokens

# Confidences come from params.py and are learned: inv.exact, inv.oldest_first, inv.event_hint,
# person.covered, plus the plan / itemised ones set by levels 3 and 4.
DUPLICATE_WINDOW = timedelta(minutes=10)



@dataclass
class AllocationResult:
    links: list[PaymentLink] = field(default_factory=list)
    unmatched: list[UnmatchedAmount] = field(default_factory=list)
    # person -> [(invoice id, event id, cents still owed)] after everything was applied
    remaining: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    plans: dict[str, Plan] = field(default_factory=dict)  # memo-explained payments, as applied


def _event_hint(memo: str, events: list[Event]) -> str | None:
    """The one event whose name shares a distinctive word with the memo, if exactly one.
    A memo that itemises several amounts ("NY dinner 107.05, festival dinner 107.04, ...")
    spans several bills, so it names no single event."""
    if len(re.findall(r"\d+(?:\.\d+)?", memo)) >= 2:
        return None
    words = {t for t in tokens(memo) if len(t) >= 4}
    hits = [e.id for e in events if words & {t for t in tokens(e.name) if len(t) >= 4}]
    return hits[0] if len(hits) == 1 else None


def allocate(
    invoices: list[Invoice],
    payments: list[Payment],
    identities: dict[str, Identity],
    directory: Directory,
    events: list[Event],
    plans: dict[str, Plan] | None = None,
    llm=None,
    ignored: set[str] | frozenset = frozenset(),
    accepted: set[str] | frozenset = frozenset(),
) -> AllocationResult:
    plans = dict(plans or {})  # human plans in; memo plans are added as they are found
    inv_by = {i.id: i for i in invoices}
    event_name = {e.id: e.name for e in events}
    text_of = {i.id: f"{i.vendor} {i.description} {event_name.get(i.event_id, '')}" for i in invoices}
    # rows[person] = [[invoice date, invoice id, cents still owed, event id], ...] oldest first
    rows: dict[str, list[list]] = {}
    for inv in invoices:
        for pid, cents in compute_obligations(inv).owed.items():
            if cents > 0:
                rows.setdefault(pid, []).append([inv.date, inv.id, cents, inv.event_id])
    for r in rows.values():
        r.sort(key=lambda x: (x[0], x[1]))

    out = AllocationResult()
    seen: list[Payment] = []

    # A person's explicit decision about where money goes beats anything computed, including
    # earlier guesses in time order. Apply decided links FIRST so the guesses work around them.
    pinned = {pid for pid, pl in plans.items() if pl.how == "human"}
    for pay in sorted((p for p in payments if p.id in pinned), key=lambda p: (p.timestamp, p.id)):
        sender = identities[pay.id].resolved or "(not on the list)"
        applied = 0
        for s in plans[pay.id].splits:
            row = next((r for r in rows.get(s.owed_by, []) if r[1] == s.invoice_id and r[2] > 0), None)
            take = min(s.cents, row[2]) if row else 0
            if take:
                row[2] -= take
                applied += take
                out.links.append(PaymentLink(pay.id, sender, s.invoice_id, s.owed_by, take, 1.0, 1.0, "human", "inv.human"))
        if applied < pay.amount_cents:
            out.unmatched.append(UnmatchedAmount(pay.id, pay.amount_cents - applied,
                                                 "reviewed: part of the decided amount matched nothing open", sender))
        out.plans[pay.id] = plans[pay.id]

    def hold(pay, why, payer=None, cents=None):
        out.unmatched.append(UnmatchedAmount(pay.id, pay.amount_cents if cents is None else cents, why, payer))

    def link(pay, payer, owed_by, row, cents, how, person_conf, inv_conf, basis):
        row[2] -= cents
        out.links.append(PaymentLink(pay.id, payer, row[1], owed_by, cents, person_conf, inv_conf, how, basis))

    for pay in sorted(payments, key=lambda p: (p.timestamp, p.id)):
        ident = identities[pay.id]
        if pay.id in pinned:
            seen.append(pay)
            continue  # already placed exactly as a person decided
        twin = next(
            (q for q in seen
             if q.amount_cents == pay.amount_cents
             and normalize(q.sender_name) == normalize(pay.sender_name)
             and pay.timestamp - q.timestamp <= DUPLICATE_WINDOW),
            None,
        )
        seen.append(pay)
        if pay.id in ignored:
            hold(pay, "reviewed: set aside by a person", ident.resolved)
            continue
        if pay.amount_cents <= 0:
            hold(pay, "refund or zero amount; for review", ident.resolved)
            continue
        if twin and not SAYS_SEPARATE.search(pay.memo) and pay.id not in plans:
            kind = "bank notification of" if BANK_ALERT.search(pay.memo) else "possible duplicate of"
            hold(pay, f"{kind} {twin.id}")
            continue
        if NOT_INCOMING.search(pay.memo) and pay.id not in plans:
            hold(pay, "not incoming money", ident.resolved)
            continue
        payer = ident.resolved
        human_plan = pay.id in plans and plans[pay.id].how == "human"
        if payer is None and not human_plan:
            hold(pay, "reviewed: sender is not on the list" if ident.method == "human"
                 else "sender not identified; for review")
            continue
        payer = payer or "(not on the list)"  # a person said who this money is for

        if payer != "(not on the list)" and pay.id not in plans and memo_names_others(pay.memo, payer, directory):
            view: dict[str, dict[str, int]] = {}
            for person, rs in rows.items():
                for r in rs:
                    if r[2] > 0:
                        view.setdefault(r[1], {})[person] = r[2]
            found = plan_memo_payment(pay, payer, directory, view, inv_by, llm)
            if found:
                plans[pay.id] = out.plans[pay.id] = found
        if pay.id in plans:  # memo explained it and the arithmetic confirmed it
            plan, applied = plans[pay.id], 0
            for s in plan.splits:
                row = next((r for r in rows.get(s.owed_by, []) if r[1] == s.invoice_id), None)
                take = min(s.cents, row[2]) if row else 0
                if take:
                    pc = (ident.confidence if s.owed_by == payer or plan.how == "human"
                          else min(ident.confidence, P.get("person.covered")))
                    link(pay, payer, s.owed_by, row, take, plan.how, pc, plan.invoice_confidence, plan.basis)
                    applied += take
            rows = {p: [r for r in v if r[2] > 0] for p, v in rows.items()}
            if applied < pay.amount_cents:
                hold(pay, "surplus: plan could not be fully applied", payer, pay.amount_cents - applied)
            continue
        if memo_names_others(pay.memo, payer, directory):
            hold(pay, "memo names another person; nothing adds up; for review", payer)
            continue

        mine = rows.get(payer, [])
        left = pay.amount_cents

        def fill(pool, how, basis, exact_basis):
            """Exact amount first (n equal shares: confidence / n), else oldest first."""
            nonlocal left
            same = [r for r in pool if r[2] == left]
            if same:
                link(pay, payer, payer, same[0], left, "exact", ident.confidence,
                     P.get(exact_basis) / len(same), exact_basis)
                left = 0
                return
            for r in pool:
                if left == 0:
                    break
                take = min(r[2], left)
                link(pay, payer, payer, r, take, how, ident.confidence, P.get(basis), basis)
                left -= take

        # level 4: the memo's amounts / words choose which invoices
        hint = _event_hint(pay.memo, events)
        # later-dated invoices are allowed too (a deposit can precede its invoice); the memo
        # must still point at them and the sums must add up
        pick = choose_items(pay, [r for r in mine if r[2] > 0], text_of, llm, pay.timestamp.date())
        pick_how = "itemised"
        if not pick:  # no memo help: does the amount equal the exact sum of some whole shares?
            pick = exact_subset(pay, [r for r in mine if r[2] > 0 and (r[0] <= pay.timestamp.date() or r[3] == hint)])
            pick_how = "whole shares"
        if pick:
            for r in pick.rows:
                take = min(r[2], left)
                if take:
                    link(pay, payer, payer, r, take, pick_how, ident.confidence, pick.confidence, pick.basis)
                    left -= take
            if 0 < left <= SUM_TOLERANCE * len(pick.rows):  # rounding in the shares, not real money
                hold(pay, "surplus: rounding after itemised pick", payer, left)
                left = 0
        if left and hint:
            fill([r for r in mine if r[3] == hint], "oldest first (event hint)",
                 "inv.event_hint", "inv.event_hint")
        if left:
            fill([r for r in mine if r[0] <= pay.timestamp.date() and r[2] > 0],
                 "oldest first", "inv.oldest_first", "inv.exact")
        rows[payer] = [r for r in mine if r[2] > 0]
        if left:
            applied = pay.amount_cents - left
            why = "surplus: more than the sender owes so far" if applied else "nothing open to apply it to"
            if pay.id in accepted:
                why = "reviewed: overpayment kept as a credit"
            hold(pay, why, payer, left)

    out.remaining = {p: [(r[1], r[3], r[2]) for r in v if r[2] > 0] for p, v in rows.items()}
    return out
