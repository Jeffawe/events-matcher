"""The human side: work through what the system could not settle.

    python3 matchpay.py review            # payments the pipeline held, one at a time
    python3 matchpay.py review --aliases  # name aliases the pipeline proposed, approve or reject
    python3 matchpay.py review --list     # just show the queue
    python3 matchpay.py review --apply decisions.json   # non-interactive

Every decision is appended to state/reviews.jsonl and honoured on the next run (a person's
word beats anything computed). Choosing a sender also offers to remember the sender's bank
name as a trusted alias, so the same name is recognised automatically next time. Decisions
are also labels the system learns its numbers from (python3 matchpay.py learn).
"""

import json
from dataclasses import dataclass, field

from models import Payment

from .dictionary import Dictionary, Source
from .reviews import Review, ReviewStore

ROUNDING_CENTS = 100  # leftovers under $1 are rounding, not worth a person's time


@dataclass
class Choice:
    label: str
    review: Review


@dataclass
class Item:
    payment: Payment
    cents: int
    reason: str
    kind: str  # sender | surplus | covering | duplicate | outgoing | other
    payer: str | None
    choices: list[Choice] = field(default_factory=list)


def _kind(reason: str) -> str:
    r = reason.lower()
    if r.startswith("sender not identified"):
        return "sender"
    if r.startswith(("bank notification", "possible duplicate")):
        return "duplicate"
    if r.startswith(("not incoming", "refund")):
        return "outgoing"
    if r.startswith("memo names"):
        return "covering"
    if r.startswith(("surplus", "nothing open")):
        return "surplus"
    return "other"


def build_queue(result, detail, payments, directory, reviews: dict[str, Review]) -> list[Item]:
    by_id = {p.id: p for p in payments}
    suggestions: dict[str, list] = {}
    for s in detail.reconciliation.suggestions:
        suggestions.setdefault(s.payment_id, []).append(s)
    links_of: dict[str, list] = {}
    for l in result.links:
        links_of.setdefault(l.payment_id, []).append(l)
    queue = []
    for u in result.unmatched:
        if u.amount_cents < ROUNDING_CENTS or u.reason.startswith("reviewed:"):
            continue
        pay = by_id[u.payment_id]
        item = Item(pay, u.amount_cents, u.reason, _kind(u.reason), u.payer_id)
        if item.kind == "sender":
            ident = detail.identities[pay.id]
            for pid, prob in sorted(ident.candidates.items(), key=lambda x: -x[1]):
                if prob >= 0.1:
                    item.choices.append(Choice(f"{directory.display[pid]} ({pid}), {prob:.0%} likely",
                                               Review(pay.id, payer=pid)))
        for s in suggestions.get(pay.id, []):
            existing = [{"invoice": l.invoice_id, "owed_by": l.owed_by, "cents": l.amount_cents}
                        for l in links_of.get(pay.id, [])]
            cents = min(s.cents, s.open_cents)
            item.choices.append(Choice(
                f"put ${cents / 100:.2f} on {directory.display[s.person_id]}'s share of {s.invoice_id} "
                f"(they still owe ${s.open_cents / 100:.2f}; {'; '.join(s.why)})",
                Review(pay.id, payer=u.payer_id,
                       links=existing + [{"invoice": s.invoice_id, "owed_by": s.person_id, "cents": cents}])))
        queue.append(item)
    return sorted(queue, key=lambda i: (i.payment.timestamp, i.payment.id))


def show(item: Item, n: int, total: int, out=print) -> None:
    p = item.payment
    out(f"\n[{n}/{total}] {p.id}  {p.timestamp:%Y-%m-%d}  sender {p.sender_name!r}  total ${p.amount_cents / 100:,.2f}")
    if p.memo:
        out(f"      memo: {p.memo!r}")
    out(f"      held ${item.cents / 100:,.2f}: {item.reason}")
    for i, c in enumerate(item.choices, 1):
        out(f"      {i}) {c.label}")


def parse_links(text: str, invoices: set[str], people: set[str]) -> list[dict] | str:
    """'inv:person:amount, inv:person:amount' -> links, or an error message."""
    links = []
    for part in [x for x in text.replace(";", ",").split(",") if x.strip()]:
        bits = [b.strip() for b in part.split(":")]
        if len(bits) != 3:
            return f"'{part.strip()}' is not invoice:person:amount"
        inv, who, amt = bits
        if inv not in invoices:
            return f"no invoice '{inv}'"
        if who not in people:
            return f"no person '{who}'"
        try:
            links.append({"invoice": inv, "owed_by": who, "cents": round(float(amt) * 100)})
        except ValueError:
            return f"'{amt}' is not an amount"
    return links or "nothing entered"


def remember_alias(dictionary: Dictionary, sender_name: str, person: str, reviewer: str, payment_id: str) -> None:
    src = Source("human", reviewer)
    fact = dictionary.find("alias", sender_name, person)
    if fact:
        dictionary.decide(fact, "trusted", src, "confirmed in review")
    else:
        dictionary.propose("alias", sender_name, person, src, "confirmed in review", [payment_id])


def review_payments(queue, store: ReviewStore, dictionary: Dictionary, directory, invoices: set[str],
                    reviewer: str = "reviewer", ask=input, out=print) -> int:
    done = 0
    people = set(directory.display)
    for n, item in enumerate(queue, 1):
        show(item, n, len(queue), out)
        while True:
            hint = ("number = that choice | p <person> = it's that person | n = nobody on the list | "
                    "m = enter links | i = set aside | s = skip | q = quit") if item.kind == "sender" else \
                   ("number = that choice | " + ("k = keep as credit | " if item.kind == "surplus" else "")
                    + "m = enter links | i = set aside | s = skip | q = quit")
            try:
                answer = ask(f"      > {hint}\n      ").strip()
            except EOFError:
                return done
            cmd = answer.lower()
            review = None
            if cmd in ("q", "quit"):
                return done
            if cmd in ("s", "skip", ""):
                break
            if cmd.isdigit() and 1 <= int(cmd) <= len(item.choices):
                review = item.choices[int(cmd) - 1].review
            elif cmd in ("i", "ignore"):
                review = Review(item.payment.id, ignore=True)
            elif cmd in ("k", "keep") and item.kind == "surplus":
                review = Review(item.payment.id, payer=item.payer, accept=True)
            elif cmd == "n" and item.kind == "sender":
                review = Review(item.payment.id, payer="")
            elif cmd.startswith("p ") and answer[2:].strip() in people:
                review = Review(item.payment.id, payer=answer[2:].strip())
            elif cmd == "m":
                parsed = parse_links(ask("      links as invoice:person:amount, ... > "), invoices, people)
                if isinstance(parsed, str):
                    out(f"      {parsed}")
                    continue
                review = Review(item.payment.id, payer=item.payer or "", links=parsed)  # "" = sender not on the list
            else:
                out("      not understood")
                continue
            review.by = reviewer
            store.add(review)
            done += 1
            out("      recorded.")
            if review.payer and item.kind == "sender":
                if ask(f"      remember {item.payment.sender_name!r} as {review.payer} next time? [Y/n] ").strip().lower() != "n":
                    remember_alias(dictionary, item.payment.sender_name, review.payer, reviewer, item.payment.id)
            break
    return done


def review_aliases(dictionary: Dictionary, directory, reviewer: str = "reviewer", ask=input, out=print) -> int:
    pending = [f for f in dictionary.facts.values() if f.status == "proposed" and f.kind == "alias"]
    out(f"{len(pending)} proposed alias(es)")
    done = 0
    for f in pending:
        out(f"\n  {f.key!r}  ->  {directory.display.get(f.person_id, f.person_id)} ({f.person_id})"
            f"\n  from {f.source.kind} {f.source.name}; {f.reason}")
        try:
            a = ask("  approve? [y]es / [n]o / [s]kip > ").strip().lower()
        except EOFError:
            break
        if a in ("y", "yes"):
            dictionary.decide(f.id, "trusted", Source("human", reviewer), "approved in review")
            done += 1
        elif a in ("n", "no"):
            dictionary.decide(f.id, "rejected", Source("human", reviewer), "rejected in review")
            done += 1
    return done


def apply_file(path: str, store: ReviewStore, dictionary: Dictionary, reviewer: str = "batch") -> int:
    """decisions.json: [{"payment_id": "p037", "payer": "mei", "remember": true}, {"payment_id": "p013", "ignore": true}, ...]"""
    rows = json.loads(open(path, encoding="utf-8").read())
    for r in rows:
        store.add(Review(r["payment_id"], r.get("payer"), r.get("links"), bool(r.get("ignore")), bool(r.get("accept")),
                         r.get("note", ""), reviewer))
        if r.get("remember") and r.get("payer") and r.get("sender_name"):
            remember_alias(dictionary, r["sender_name"], r["payer"], reviewer, r["payment_id"])
    return len(rows)
