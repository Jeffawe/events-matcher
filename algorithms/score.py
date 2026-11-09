"""Score the pipeline against scenarios/expected.json.

    python3 -m algorithms.score            # rules only
    python3 -m algorithms.score --llm      # plus the small local model (Ollama)
    python3 -m algorithms.score -v         # list every keyed payment

Three questions, kept apart:
  IDENTITY  did we name the right sender?  Wrong resolutions must be ~0; uncertain is fine.
  PERSON    did the money land on the right person's obligations?
  INVOICE   did it land on the key's exact invoice?  Provisional; a later level confirms.
The answer key covers p001-p073 only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.llm_ollama import OllamaVoter  # noqa: E402
from algorithms.pipeline import run_pipeline  # noqa: E402
from run import DATA, load_scenarios, to_cents  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--data", default=str(DATA), help="scenario folder (default: scenarios/)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    events, invoices, payments = load_scenarios(data)
    llm = OllamaVoter() if args.llm else None
    res = run_pipeline(events, invoices, payments, llm=llm)
    key = {p["payment_id"]: p for p in json.loads((data / "expected.json").read_text())["payments"]}
    by_id = {p.id: p for p in payments}

    # ---------- identity ----------
    tally, wrong_ids, rows = Counter(), [], []
    for pid, k in key.items():
        ident = res.identities[pid]
        truth = k["payer"] or k.get("true_payer")  # held payments may still name who it was for
        if ident.resolved is None:
            outcome = "uncertain (key held it)" if k["payer"] is None else "uncertain (key knew)"
        elif k["payer"] is not None:
            outcome = "resolved right" if ident.resolved == k["payer"] else "RESOLVED WRONG"
        else:
            outcome = "resolved, key held: matches its true_payer" if ident.resolved == truth else "resolved, key held: DIFFERS"
        tally[outcome] += 1
        if "WRONG" in outcome or "DIFFERS" in outcome:
            wrong_ids.append(pid)
        rows.append((pid, by_id[pid].sender_name, ident, k, outcome))
    print("IDENTITY (payments with a key:", len(key), ")")
    for outcome, n in sorted(tally.items()):
        print(f"  {n:>3}  {outcome}")
    methods = Counter(i.method for i in res.identities.values())
    print("  methods over all 100 payments:", dict(methods))
    if llm:
        print(f"  llm model {llm.model}: {llm.calls} new calls, {llm.hits} from cache, reachable={llm.up}")
    for pid in wrong_ids:
        _, sender, ident, k, outcome = next(r for r in rows if r[0] == pid)
        print(f"    !! {pid} '{sender}' -> {ident.resolved} ({ident.confidence:.0%}, {ident.method}); key payer={k['payer']} true={k.get('true_payer')}")

    # calibration of identity probability
    buckets: dict[str, list[bool]] = {}
    for pid, sender, ident, k, outcome in rows:
        if ident.resolved and k["payer"] is not None and ident.method != "exact":
            b = "≥0.99" if ident.confidence >= 0.99 else "0.95-0.99" if ident.confidence >= 0.95 else "0.90-0.95"
            buckets.setdefault(b, []).append(ident.resolved == k["payer"])
    for b, hits in sorted(buckets.items()):
        print(f"  level-2 confidence {b}: {sum(hits)}/{len(hits)} right")

    # ---------- allocation ----------
    want = {pid: {(l["invoice"], l["owed_by"], to_cents(l["amount"])) for l in k["links"]} for pid, k in key.items()}
    keyed = [l for l in res.allocation.links if l.payment_id in key]
    held_by_key = [l for l in keyed if not want[l.payment_id]]  # key held it; we placed it anyway
    keyed = [l for l in keyed if want[l.payment_id]]
    person_ok = [l for l in keyed if l.owed_by in {w[1] for w in want[l.payment_id]}]
    inv_ok = [l for l in keyed if (l.invoice_id, l.owed_by, l.amount_cents) in want[l.payment_id]]
    print(f"\nALLOCATION on keyed payments: {len(keyed)} links")
    print(f"  PERSON  right: {len(person_ok)}   wrong: {len(keyed) - len(person_ok)}")
    if held_by_key:
        print(f"  ({len(held_by_key)} more links on payments the key held, not scored: {sorted({l.payment_id for l in held_by_key})})")
    print(f"  INVOICE exact: {len(inv_ok)} of {len(keyed)}   (rest provisional)")
    by_conf: dict[float, list[bool]] = {}
    for l in keyed:
        ok = (l.invoice_id, l.owed_by, l.amount_cents) in want[l.payment_id]
        by_conf.setdefault(round(l.invoice_confidence, 2), []).append(ok)
    for c in sorted(by_conf, reverse=True):
        h = by_conf[c]
        print(f"    invoice_confidence {c:>4}: {sum(h)}/{len(h)} = {sum(h) / len(h):.0%}")
    placed = {l.payment_id for l in keyed}
    got = sum(l.amount_cents for l in keyed)
    should = sum(to_cents(x["amount"]) for pid in placed for x in key[pid]["links"])
    print(f"  money placed ${got / 100:,.2f}; the key settles ${should / 100:,.2f} for those payments")
    print("  held, all 100 payments:", dict(Counter(u.reason.split(":")[0] for u in res.allocation.unmatched)))
    print(f"  alias proposals written to the dictionary (for human approval): {res.proposals}")

    # ---------- invoice statuses ----------
    kinv = json.loads((data / "expected.json").read_text()).get("invoices")
    if kinv:
        from matcher import invoice_statuses
        ours = {s.invoice_id: s for s in invoice_statuses(invoices, res.allocation.links)}
        same_status = same_paid = 0
        for kv in kinv:
            s = ours[kv["invoice"]]
            same_status += str(s.status) == kv["status"]
            same_paid += {x.person_id: x.paid_cents for x in s.shares} == {x["person"]: to_cents(x["paid"]) for x in kv["shares"]}
        print(f"\nINVOICE STATUS vs key: status matches on {same_status} of {len(kinv)} invoices; "
              f"every share's paid amount matches on {same_paid} of {len(kinv)}")

    # ---------- level 3 ----------
    print("\nLEVEL 3")
    for pid in sorted(k for k, v in key.items() if v["how"] == "memo"):
        got_links = {(l.invoice_id, l.owed_by, l.amount_cents) for l in res.allocation.links if l.payment_id == pid}
        print(f"  3a {pid} {'identical to key' if got_links == want[pid] else 'DIFFERS from key':<17} plan={'yes' if pid in res.plans else 'no'}")
    for pid, plan in sorted(res.plans.items()):
        if pid not in key:
            print(f"  3a {pid} (no key) {[(s.owed_by, s.invoice_id, s.cents / 100) for s in plan.splits]}")
    hinted = sorted({l.payment_id for l in res.allocation.links if "event hint" in l.how})
    for pid in hinted:
        got_links = {(l.invoice_id, l.owed_by, l.amount_cents) for l in res.allocation.links if l.payment_id == pid}
        status = "identical to key" if pid in key and got_links == want[pid] else ("DIFFERS from key" if pid in key else "no key")
        print(f"  3c event hint {pid} {by_id[pid].memo!r}: {status}")
    print("\nLEVEL 4 (itemised memos choose the invoices)")
    items = sorted({l.payment_id for l in res.allocation.links if l.how == "itemised"})
    for pid in items:
        mine = [l for l in res.allocation.links if l.payment_id == pid]
        got_links = {(l.invoice_id, l.owed_by, l.amount_cents) for l in mine}
        status = ("identical to key" if got_links == want[pid] else "DIFFERS from key") if pid in key else "no key"
        print(f"  {pid} {by_id[pid].memo[:44]!r:<48} {status}: " + ", ".join(f"{l.invoice_id} {l.amount_cents / 100:.2f}@{l.invoice_confidence}" for l in mine))
    print(f"  {len(items)} payments placed by memo content")
    for u in sorted(res.allocation.unmatched, key=lambda u: u.payment_id):
        if u.reason.startswith(("bank notification", "possible duplicate", "not incoming")):
            k = key.get(u.payment_id)
            print(f"  3c held {u.payment_id} [{u.reason}]  key: {k['how'] + ' / ' + k.get('reason', '') if k else 'no key'}")
    rec = res.reconciliation
    open_people = [b for b in rec.balances if b.paid_cents < b.owed_cents]
    print(f"  3b {len(rec.balances)} person-event balances; {len(open_people)} not fully paid, ${sum(b.owed_cents - b.paid_cents for b in open_people) / 100:,.2f} still open")
    for sg in rec.suggestions:
        print(f"  3b suggest {sg.payment_id} ({sg.payer_id}, ${sg.cents / 100:.2f}) -> {sg.person_id} on {sg.invoice_id} (open ${sg.open_cents / 100:.2f}) score {sg.score}: {'; '.join(sg.why)}")
    print(f"  3b rounding-size leftovers (<$1): ${rec.rounding_cents / 100:.2f}")
    print(f"  3b surplus with no plausible home: ${rec.unexplained_cents / 100:,.2f}")

    if any("noise" in k for k in key.values()):  # synthetic data says what mess each payment got
        print("\nBY KIND OF NAME MESS (senders the generator knows)")
        for kind in sorted({k["noise"] for k in key.values()}):
            c = Counter()
            for pid, sender, ident, k, outcome in rows:
                if k["noise"] == kind and (k["payer"] or k.get("true_payer")):
                    c["right" if "right" in outcome or "matches" in outcome else "WRONG" if ("WRONG" in outcome or "DIFFERS" in outcome) else "uncertain"] += 1
            print(f"  {kind:<14} right {c['right']:>3}   uncertain {c['uncertain']:>3}   WRONG {c['WRONG']:>3}")
        print("\nBY PAYMENT HABIT (links expected / on the right person / on the exact invoice)")
        for habit in sorted({k["pattern"] for k in key.values()}):
            ids = {pid for pid, k in key.items() if k["pattern"] == habit}
            exp = sum(len(want[i]) for i in ids)
            mine = [l for l in res.allocation.links if l.payment_id in ids]
            pe = sum(1 for l in mine if l.owed_by in {w[1] for w in want[l.payment_id]})
            ie = sum(1 for l in mine if (l.invoice_id, l.owed_by, l.amount_cents) in want[l.payment_id])
            print(f"  {habit:<14} payments {len(ids):>3}   links expected {exp:>3}   placed {len(mine):>3}   person-right {pe:>3}   invoice-exact {ie:>3}")

    if args.verbose:
        print("\nPAYMENT-BY-PAYMENT")
        for pid, sender, ident, k, outcome in rows:
            top = ", ".join(f"{p} {v:.0%}" for p, v in sorted(ident.candidates.items(), key=lambda x: -x[1])[:3])
            print(f"  {pid} {sender!r:<22} key={k['payer']!s:<9} -> {ident.resolved!s:<9} [{ident.method:<11}] {top:<28} {outcome}")
    unkeyed = [i for pid, i in sorted(res.identities.items()) if pid not in key]
    print(f"\nNO ANSWER KEY (p074+), eyeball: {sum(1 for i in unkeyed if i.resolved)} resolved, {sum(1 for i in unkeyed if not i.resolved)} uncertain")
    if args.verbose:
        for i in unkeyed:
            top = ", ".join(f"{p} {v:.0%}" for p, v in sorted(i.candidates.items(), key=lambda x: -x[1])[:3])
            print(f"  {i.payment_id} {by_id[i.payment_id].sender_name!r:<22} -> {i.resolved!s:<9} [{i.method}] {top}")
    return len(wrong_ids)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
