"""Glue for the in-browser demo (index.html, running this code in Pyodide). Not used by the
command line. Each function runs the real pipeline on a scenario folder and returns plain
text (JSON) that the page can display and offer for download.
"""

import json
import tempfile
from collections import Counter
from pathlib import Path


def _score(root: Path, result, detail) -> dict | None:
    """If the scenario has an answer key, say how the run compares to it."""
    from run import to_cents

    path = root / "expected.json"
    if not path.exists():
        return None
    key = {p["payment_id"]: p for p in json.loads(path.read_text(encoding="utf-8"))["payments"]}
    ident = Counter()
    for pid, k in key.items():
        if k["payer"] is None or pid not in detail.identities:
            continue
        got = detail.identities[pid].resolved
        ident["uncertain" if got is None else "right" if got == k["payer"] else "wrong"] += 1
    want = {pid: {(l["invoice"], l["owed_by"], to_cents(l["amount"])) for l in k["links"]} for pid, k in key.items()}
    keyed = [l for l in result.links if want.get(l.payment_id)]
    person_ok = sum(l.owed_by in {w[1] for w in want[l.payment_id]} for l in keyed)
    exact = sum((l.invoice_id, l.owed_by, l.amount_cents) in want[l.payment_id] for l in keyed)
    return {
        "senders_right": ident["right"], "senders_wrong": ident["wrong"], "senders_uncertain": ident["uncertain"],
        "links": len(keyed), "links_right_person": person_ok, "links_exact_invoice": exact,
        "money_placed_cents": sum(l.amount_cents for l in keyed),
        "money_in_key_cents": sum(c for pid in {l.payment_id for l in keyed} for _, _, c in want[pid]),
    }


def _run(root: Path, title: str) -> str:
    from algorithms.identity import Directory
    from algorithms.report import write_outputs
    from algorithms.review import ROUNDING_CENTS
    from matcher import match_detailed
    from run import load_scenarios

    events, invoices, payments = load_scenarios(root)
    result, detail = match_detailed(events, invoices, payments)  # in memory: nothing is saved
    out = Path(tempfile.mkdtemp())
    write_outputs(out, title, result, detail, events, invoices, payments, Directory.build(events))
    needs = [u for u in result.unmatched if u.amount_cents >= ROUNDING_CENTS]
    return json.dumps({
        "summary": {
            "payments": len(payments), "invoices": len(invoices), "events": len(events), "links": len(result.links),
            "placed_cents": sum(l.amount_cents for l in result.links),
            "held_cents": sum(u.amount_cents for u in needs), "held_items": len(needs),
            "invoices_paid": sum(1 for i in result.invoices if str(i.status) == "paid"),
            "senders_identified": sum(1 for i in detail.identities.values() if i.resolved),
        },
        "score": _score(root, result, detail),
        "files": {name: (out / name).read_text(encoding="utf-8") for name in ("report.html", "result.json", "links.csv", "held.csv")},
    })


def run_folder(path: str, title: str) -> str:
    return _run(Path(path), title)


def generate_and_run(seed: int, events: int = 4, people: int = 11) -> str:
    from algorithms.synth import generate

    root = Path(tempfile.mkdtemp())
    generate(int(seed), root, int(events), int(people))
    return _run(root, f"Generated world, seed {int(seed)}")


def run_custom(files_json: str) -> str:
    """files_json: {"events/NAME.json": text, "payments.json": text, "expected.json": text (optional)}"""
    root = Path(tempfile.mkdtemp())
    for rel, text in json.loads(files_json).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return _run(root, "Your data")
