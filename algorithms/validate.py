"""Check a scenario folder before running it, and say what is wrong in plain words.

    errors, warnings = validate(Path("scenarios"))

Errors stop a run (the loader would crash or give nonsense). Warnings are things that are
legal but worth knowing, such as a split that adds to 90% or a name that changes between
events. The format itself is described in data-format.html and the README.
"""

import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

MAX_SHOWN = 25


def _number(value, allow_zero=True):
    """A Decimal if value is a plain number (or a numeric string), else None. Booleans are not numbers."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        d = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _text(value) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _load(path: Path, problems: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        problems.append(f"{path.name}: not valid JSON ({e.msg}, line {e.lineno}, column {e.colno}).")
    except OSError as e:
        problems.append(f"{path.name}: could not be read ({e}).")
    return None


def validate(data: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_invoice: dict[str, str] = {}
    person_names: dict[str, tuple[str, str]] = {}  # person id -> (name, event) first seen

    event_files = sorted((data / "events").glob("*.json")) if (data / "events").is_dir() else []
    if not event_files:
        errors.append("No event files found. Put one JSON file per event in an events/ folder.")
    for path in event_files:
        ev = _load(path, errors)
        if ev is None:
            continue
        where = path.name
        if not isinstance(ev, dict):
            errors.append(f"{where}: an event file must be a JSON object, not a list.")
            continue
        for key in ("id", "name"):
            if not _text(ev.get(key)):
                errors.append(f'{where}: missing "{key}" (text).')
        for key in ("start_date", "end_date"):
            try:
                date.fromisoformat(str(ev.get(key)))
            except ValueError:
                errors.append(f'{where}: "{key}" must be a date like 2026-03-02 (got {ev.get(key)!r}).')
        people = ev.get("people")
        ids: set[str] = set()
        if not isinstance(people, list) or not people:
            errors.append(f'{where}: "people" must be a list with at least one {{"id", "name"}}.')
            people = []
        for i, p in enumerate(people, 1):
            if not isinstance(p, dict) or not _text(p.get("id")) or not _text(p.get("name")):
                errors.append(f'{where}: person {i} needs both "id" and "name".')
                continue
            if p["id"] in ids:
                errors.append(f'{where}: person id "{p["id"]}" appears twice.')
            ids.add(p["id"])
            if p["id"] in person_names and person_names[p["id"]][0] != p["name"]:
                warnings.append(f'Person id "{p["id"]}" is "{p["name"]}" in {where} but "{person_names[p["id"]][0]}" '
                                f'in {person_names[p["id"]][1]}. Ids should mean the same person in every event.')
            person_names.setdefault(p["id"], (p["name"], where))
        invoices = ev.get("invoices")
        if not isinstance(invoices, list):
            errors.append(f'{where}: "invoices" must be a list.')
            continue
        for n, inv in enumerate(invoices, 1):
            tag = f'{where}, invoice {n}' + (f' ("{inv["id"]}")' if isinstance(inv, dict) and _text(inv.get("id")) else "")
            if not isinstance(inv, dict):
                errors.append(f"{tag}: must be an object.")
                continue
            for key in ("id", "vendor", "description"):
                if not _text(inv.get(key)):
                    errors.append(f'{tag}: missing "{key}" (text).')
            if _text(inv.get("id")):
                if inv["id"] in seen_invoice:
                    errors.append(f'{tag}: invoice id "{inv["id"]}" is also used in {seen_invoice[inv["id"]]}. Ids must be unique across all events.')
                seen_invoice[inv["id"]] = where
            try:
                date.fromisoformat(str(inv.get("date")))
            except ValueError:
                errors.append(f'{tag}: "date" must be a date like 2026-03-02 (got {inv.get("date")!r}).')
            total = _number(inv.get("total"))
            if total is None or total <= 0:
                errors.append(f'{tag}: "total" must be a positive amount in dollars, like "2000.00" (got {inv.get("total")!r}).')
            shares = inv.get("shares")
            if not isinstance(shares, list) or not shares:
                errors.append(f'{tag}: "shares" must list who pays what.')
                continue
            pct = amt = Decimal(0)
            for s in shares:
                if not isinstance(s, dict) or s.get("person") not in ids:
                    errors.append(f'{tag}: a share names person "{s.get("person") if isinstance(s, dict) else s}", who is not in this event\'s people.')
                    continue
                has_p, has_a = "percent" in s, "amount" in s
                if has_p == has_a:
                    errors.append(f'{tag}: the share for "{s["person"]}" needs exactly one of "percent" or "amount".')
                    continue
                v = _number(s["percent" if has_p else "amount"])
                if v is None or v < 0:
                    errors.append(f'{tag}: the share for "{s["person"]}" has a bad {"percent" if has_p else "amount"} ({s["percent" if has_p else "amount"]!r}).')
                    continue
                pct, amt = (pct + v, amt) if has_p else (pct, amt + v)
            if total and total > 0 and not any(e.startswith(tag) for e in errors):
                covered = (pct / 100) * total + amt
                if abs(covered - total) >= Decimal("0.005"):
                    word = "over-assigned" if covered > total else "only partly assigned"
                    warnings.append(f"{tag}: shares cover ${covered:,.2f} of ${total:,.2f} ({word}). The report will flag it.")

    pay_path = data / "payments.json"
    pays = _load(pay_path, errors) if pay_path.exists() else None
    if not pay_path.exists():
        errors.append("payments.json is missing.")
    elif pays is not None:
        if not isinstance(pays, list) or not pays:
            errors.append("payments.json must be a non-empty list of payments.")
        else:
            seen_pay: set[str] = set()
            for n, p in enumerate(pays, 1):
                tag = f"payments.json, payment {n}" + (f' ("{p["id"]}")' if isinstance(p, dict) and _text(p.get("id")) else "")
                if not isinstance(p, dict):
                    errors.append(f"{tag}: must be an object.")
                    continue
                for key in ("id", "sender_name"):
                    if not _text(p.get(key)):
                        errors.append(f'{tag}: missing "{key}" (text).')
                if _text(p.get("id")):
                    if p["id"] in seen_pay:
                        errors.append(f'{tag}: payment id "{p["id"]}" appears twice.')
                    seen_pay.add(p["id"])
                amount = _number(p.get("amount"))
                if amount is None:
                    errors.append(f'{tag}: "amount" must be a number of dollars, like "200.00" (got {p.get("amount")!r}).')
                elif amount <= 0:
                    warnings.append(f"{tag}: amount is {amount}. Zero and negative amounts are held for review, not matched.")
                try:
                    datetime.fromisoformat(str(p.get("timestamp")))
                except ValueError:
                    errors.append(f'{tag}: "timestamp" must look like 2027-05-10T09:12:00 (got {p.get("timestamp")!r}).')
                if "memo" in p and p["memo"] is not None and not isinstance(p["memo"], str):
                    errors.append(f'{tag}: "memo" must be text.')

    def cap(items: list[str], what: str) -> list[str]:
        return items if len(items) <= MAX_SHOWN else items[:MAX_SHOWN] + [f"...and {len(items) - MAX_SHOWN} more {what}."]

    return cap(errors, "problems"), cap(warnings, "notes")
