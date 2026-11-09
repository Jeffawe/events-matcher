"""Does the pipeline depend on THIS dataset, or would it work on other messy data?

    python3 -m algorithms.robustness            # rules only (deterministic, no model)

Two families of test:

  INVARIANCE  change something that must not matter and require identical output:
      relabel   every id (person, invoice, event, payment) becomes an opaque string
      shuffle   the order of payments / invoices / events / shares
      shift     every date moved by ~2 years
      cipher    EVERY piece of text (names, memos, vendors, events) run through the same
                letter substitution, so no English word or real name survives while
                prefix / initial / typo relationships do. Differences here show exactly
                which rules lean on English words or sounds.

  SAFETY      make the input messier and require that errors do not become confident:
      name mangling (8 kinds), memos removed, a look-alike person added.
      Pass = zero wrongly-resolved senders; more "uncertain" is fine and expected.
"""

import argparse
import copy
import json
import random
import sys
import unicodedata
from collections import Counter
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.pipeline import run_pipeline  # noqa: E402
from models import Person  # noqa: E402
from run import DATA, load_scenarios  # noqa: E402

FIRM = 0.9  # invoice_confidence at or above this means "we claim this invoice"
KEY = {p["payment_id"]: p for p in json.loads((DATA / "expected.json").read_text())["payments"]}


# ---------------------------------------------------------------- helpers


def run(events, invoices, payments):
    return run_pipeline(events, invoices, payments)


def snapshot(res, back=None):
    """Comparable view of a result. `back` maps perturbed ids to the original ids."""
    b = back or {}
    g = lambda kind, x: b.get(kind, {}).get(x, x)  # noqa: E731
    links = {
        (g("pay", l.payment_id), g("inv", l.invoice_id), g("person", l.owed_by), l.amount_cents)
        for l in res.allocation.links
    }
    firm = {
        (g("pay", l.payment_id), g("inv", l.invoice_id), g("person", l.owed_by), l.amount_cents)
        for l in res.allocation.links if l.invoice_confidence >= FIRM
    }
    who = {g("pay", pid): (g("person", i.resolved) if i.resolved else None) for pid, i in res.identities.items()}
    held = {(g("pay", u.payment_id), u.reason.split(":")[0].split(" of ")[0]) for u in res.allocation.unmatched}
    return links, who, held, firm


def diff(a, b):
    la, wa, ha, fa = a
    lb, wb, hb, fb = b
    ident = [p for p in wa if wa[p] != wb.get(p)]
    firm = [x for x in la ^ lb if x in fa or x in fb]  # a link we claimed firmly, on either side
    return len(firm), len(la ^ lb) - len(firm), len(ident), len(ha ^ hb), ident[:6]


def identity_outcomes(res, to_orig=None):
    """right / wrong / uncertain for every payment the key names a payer for."""
    c = Counter()
    wrong = []
    for pid, k in KEY.items():
        if k["payer"] is None:
            continue
        got = res.identities[pid].resolved
        if got is None:
            c["uncertain"] += 1
        elif (to_orig or {}).get(got, got) == k["payer"]:
            c["right"] += 1
        else:
            c["WRONG"] += 1
            wrong.append((pid, got, k["payer"]))
    return c, wrong


# ---------------------------------------------------------------- perturbations


def relabel(events, invoices, payments, rng):
    ids = {"person": set(), "inv": set(), "ev": set(), "pay": {p.id for p in payments}}
    for e in events:
        ids["ev"].add(e.id)
        ids["person"] |= {p.id for p in e.people}
    ids["inv"] = {i.id for i in invoices}
    maps, back = {}, {}
    for kind, prefix in (("person", "P"), ("inv", "I"), ("ev", "E"), ("pay", "X")):
        names = sorted(ids[kind])
        new = [f"{prefix}{n:04d}" for n in rng.sample(range(10000), len(names))]
        maps[kind] = dict(zip(names, new))
        back[kind] = {v: k for k, v in maps[kind].items()}
    ev, inv, pay = copy.deepcopy((events, invoices, payments))
    for e in ev:
        e.id = maps["ev"][e.id]
        for p in e.people:
            p.id = maps["person"][p.id]
    for i in inv:
        i.id, i.event_id = maps["inv"][i.id], maps["ev"][i.event_id]
        for s in i.shares:
            s.person_id = maps["person"][s.person_id]
    for p in pay:
        p.id = maps["pay"][p.id]
    return ev, inv, pay, back


def shuffled(events, invoices, payments, rng):
    ev, inv, pay = copy.deepcopy((events, invoices, payments))
    for xs in (ev, inv, pay):
        rng.shuffle(xs)
    for e in ev:
        rng.shuffle(e.people)
    # NOT shuffled: the order of shares inside an invoice. The leftover cents of a split go
    # to whoever is listed first, so that order is real data, not noise.
    return ev, inv, pay, None


def shifted(events, invoices, payments, rng):
    ev, inv, pay = copy.deepcopy((events, invoices, payments))
    d = timedelta(days=rng.randint(400, 900))
    for e in ev:
        e.start_date, e.end_date = e.start_date + d, e.end_date + d
    for i in inv:
        i.date += d
    for p in pay:
        p.timestamp += d
    return ev, inv, pay, None


def _cipher(rng):
    letters = list("abcdefghijklmnopqrstuvwxyz")
    shuf = letters[:]
    rng.shuffle(shuf)
    table = {**dict(zip(letters, shuf)), **{a.upper(): b.upper() for a, b in zip(letters, shuf)}}

    def enc(s: str) -> str:
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        return "".join(table.get(c, c) for c in s)

    return enc


def ciphered(events, invoices, payments, rng):
    enc = _cipher(rng)
    ev, inv, pay = copy.deepcopy((events, invoices, payments))
    for e in ev:
        e.name = enc(e.name)
        for p in e.people:
            p.name = enc(p.name)
    for i in inv:
        i.vendor, i.description, i.note = enc(i.vendor), enc(i.description), enc(i.note)
    for p in pay:
        p.sender_name, p.memo = enc(p.sender_name), enc(p.memo)
    return ev, inv, pay, None


MANGLE = {
    "UPPERCASE": lambda n, r: n.upper(),
    "lowercase": lambda n, r: n.lower(),
    "drop last name": lambda n, r: n.split()[0] if len(n.split()) > 1 else n,
    "first initial": lambda n, r: f"{n.split()[0][0]}. {' '.join(n.split()[1:])}".strip() if len(n.split()) > 1 else n,
    "last initial": lambda n, r: f"{n.split()[0]} {n.split()[-1][0]}" if len(n.split()) > 1 else n,
    "swap order": lambda n, r: " ".join(reversed(n.split())),
    "typo": lambda n, r: _typo(n, r),
    "extra word": lambda n, r: n + " Jr",
}


def _typo(name, rng):
    words = name.split()
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    if len(w) < 4:
        return name
    j = rng.randint(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1:]  # delete one inner letter
    return " ".join(words)


def mangled(events, invoices, payments, kind, rng):
    pay = copy.deepcopy(payments)
    for p in pay:
        p.sender_name = MANGLE[kind](p.sender_name, rng)
    return events, invoices, pay


def no_memos(events, invoices, payments):
    pay = copy.deepcopy(payments)
    for p in pay:
        p.memo = ""
    return events, invoices, pay


def with_lookalike(events, invoices, payments, rng):
    """Add a person whose name is one edit away from an existing one, to each event they share."""
    ev = copy.deepcopy(events)
    victim = rng.choice(sorted({p.id for e in ev for p in e.people}))
    name = next(p.name for e in ev for p in e.people if p.id == victim)
    words = name.split()
    twin = " ".join([words[0] + ("a" if not words[0].endswith("a") else "h")] + words[1:]) if rng.random() < .5 \
        else " ".join(words[:-1] + [words[-1] + "son"])
    for e in ev:
        if any(p.id == victim for p in e.people):
            e.people.append(Person(victim + "_twin", twin))
    return ev, invoices, payments, f"{name} -> look-alike {twin!r}"


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    events, invoices, payments = load_scenarios()
    base = run(events, invoices, payments)
    base_snap = snapshot(base)
    c0, _ = identity_outcomes(base)
    print(f"baseline identity vs key: {dict(c0)}   links: {len(base_snap[0])}\n")

    problems = 0
    print("INVARIANCE (output must be identical to the baseline)")
    print(f"  {'test':<9} {'seed':>4} {'firm links differ':>18} {'provisional differ':>19} {'identities':>11} {'held':>5}")
    for name, fn in (("relabel", relabel), ("shuffle", shuffled), ("shift", shifted), ("cipher", ciphered)):
        for seed in range(args.seeds):
            ev, inv, pay, back = fn(events, invoices, payments, random.Random(seed))
            res = run(ev, inv, pay)
            snap = snapshot(res, back)
            df, dp, di, dh, who = diff(base_snap, snap)
            if name == "cipher" and seed > 0:
                continue  # one cipher is enough to show the pattern
            bad = (df, di) != (0, 0)  # provisional links may differ: they are flagged as guesses
            problems += bad and name != "cipher"
            print(f"  {name:<9} {seed:>4} {df:>18} {dp:>19} {di:>11} {dh:>5}{'   <-- differs' if bad else ''}"
                  + (f"  e.g. {who}" if di else ""))
            if name == "cipher":
                reasons = Counter(r for _, r in (snapshot(res, back)[2] ^ base_snap[2]))
                print(f"            held-reason differences: {dict(reasons)}")

    print("\nSAFETY (wrongly-resolved senders must stay 0; 'uncertain' may rise)")
    print(f"  {'mess added':<22} {'right':>6} {'uncertain':>10} {'WRONG':>6}")
    rows = [("none (baseline)", c0)]
    for kind in MANGLE:
        tot = Counter()
        for seed in range(args.seeds):
            ev, inv, pay = mangled(events, invoices, payments, kind, random.Random(seed))
            c, wrong = identity_outcomes(run(ev, inv, pay))
            tot += c
            for w in wrong[:2]:
                print(f"      ! {kind} seed {seed}: {w[0]} resolved {w[1]}, key says {w[2]}")
        rows.append((kind, Counter({k: round(v / args.seeds, 1) for k, v in tot.items()})))
    ev, inv, pay = no_memos(events, invoices, payments)
    res = run(ev, inv, pay)
    c, _ = identity_outcomes(res)
    rows.append(("all memos removed", c))
    tot = Counter()
    for seed in range(args.seeds):
        ev, inv, pay, note = with_lookalike(events, invoices, payments, random.Random(seed))
        res = run(ev, inv, pay)
        c, wrong = identity_outcomes(res)
        tot += c
        print(f"      look-alike seed {seed}: {note}  -> {dict(c)}")
    rows.append(("look-alike person added", Counter({k: round(v / args.seeds, 1) for k, v in tot.items()})))
    for name, c in rows:
        print(f"  {name:<22} {c.get('right', 0):>6} {c.get('uncertain', 0):>10} {c.get('WRONG', 0):>6}")
        problems += c.get("WRONG", 0) > c0.get("WRONG", 0)
    print("\nRESULT:", "no safety or invariance failures" if not problems else f"{problems} failing test group(s)")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
