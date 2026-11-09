"""Loads the scenario data and passes it through the matcher.  python3 run.py

Only events/ and payments.json go into the matcher. expected.json is the
answer key and is never read here.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from matcher import match_payments
from models import Event, Invoice, Payment, Person, Share
from obligations import compute_obligations

DATA = Path(__file__).parent / "scenarios"


def to_cents(dollars: str) -> int:
    return int(Decimal(dollars) * 100)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_share(raw: dict) -> Share:
    if "percent" in raw:
        return Share(raw["person"], percent=Decimal(raw["percent"]))
    return Share(raw["person"], amount_cents=to_cents(raw["amount"]))


def load_scenarios(data: Path = DATA):
    events, invoices = [], []
    for path in sorted((data / "events").glob("*.json")):
        raw = read(path)
        events.append(
            Event(
                raw["id"],
                raw["name"],
                date.fromisoformat(raw["start_date"]),
                date.fromisoformat(raw["end_date"]),
                [Person(p["id"], p["name"]) for p in raw["people"]],
            )
        )
        invoices += [
            Invoice(
                i["id"],
                raw["id"],
                date.fromisoformat(i["date"]),
                i["vendor"],
                i["description"],
                to_cents(i["total"]),
                [load_share(s) for s in i["shares"]],
                i.get("note", ""),
            )
            for i in raw["invoices"]
        ]
    payments = [
        Payment(
            p["id"],
            p["sender_name"],
            to_cents(p["amount"]),
            datetime.fromisoformat(p["timestamp"]),
            p.get("memo", ""),
        )
        for p in read(data / "payments.json")
    ]
    return events, invoices, payments


if __name__ == "__main__":
    events, invoices, payments = load_scenarios()
    print(f"Loaded {len(events)} events, {len(invoices)} invoices, {len(payments)} payments")
    
    for inv in invoices:
        ob = compute_obligations(inv)
        flag = "" if not ob.problems else "  <-- " + "; ".join(ob.problems)
        print(f"{inv.id:<14}{inv.total_cents / 100:>9.2f}  {ob.owed}{flag}")
    # print(match_payments(events, invoices, payments))
