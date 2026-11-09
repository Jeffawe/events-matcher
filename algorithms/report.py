"""Turn a match into something a person can read: one HTML page, plus JSON and CSV.

    out/report.html   open in a browser; no scripts, no external files
    out/result.json   everything, for other programs
    out/links.csv     one row per payment-to-invoice link
    out/held.csv      one row per payment (or part) that needs a person
"""

import csv
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from models import PaidStatus

from .review import ROUNDING_CENTS, _kind

CSS = """
:root{--bg:#fafaf8;--fg:#1f2328;--mute:#656d76;--card:#fff;--line:#e3e1db;--ok:#1a7f4b;--warn:#a15c00;--bad:#c4321f;--accent:#2457c5}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e6e6e3;--mute:#9aa1a9;--card:#1f2227;--line:#32363d;--ok:#4cc38a;--warn:#e2a336;--bad:#f07a68;--accent:#7aa2f7}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
.sub{color:var(--mute);margin:0 0 20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:22px}
.card span{color:var(--mute);font-size:13px}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}th{color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.wrap{overflow-x:auto}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}.paid{color:var(--ok);border-color:var(--ok)}
.partial{color:var(--warn);border-color:var(--warn)}.unpaid{color:var(--bad);border-color:var(--bad)}.mute{color:var(--mute)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;min-width:60px}.bar i{display:block;height:100%;background:var(--ok)}
.q{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--warn);border-radius:8px;padding:10px 14px;margin:8px 0}
.q small{color:var(--mute)}details summary{cursor:pointer;color:var(--accent)}
"""


def money(c: int) -> str:
    return f"{c / 100:,.2f}"


def _tag(status) -> str:
    s = str(status)
    return f'<span class="tag {s}">{s}</span>'


def render_html(title: str, result, detail, events, invoices, payments, directory, model: str | None) -> str:
    pay = {p.id: p for p in payments}
    inv = {i.id: i for i in invoices}
    name = directory.display
    links_by: dict[str, list] = {}
    for l in result.links:
        links_by.setdefault(l.payment_id, []).append(l)
    held_by = {u.payment_id: u for u in result.unmatched}
    placed = sum(l.amount_cents for l in result.links)
    held = sum(u.amount_cents for u in result.unmatched if u.amount_cents >= ROUNDING_CENTS)
    needs = [u for u in result.unmatched if u.amount_cents >= ROUNDING_CENTS]
    statuses = [i.status for i in result.invoices]
    ident = detail.identities
    resolved = sum(1 for i in ident.values() if i.resolved)

    out = [f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{escape(title)}</title><style>{CSS}</style><main>",
           f"<h1>{escape(title)}</h1><p class='sub'>{len(payments)} payments · {len(invoices)} invoices · "
           f"{len(events)} events · local model: {escape(model) if model else 'not used'}</p>", "<div class='cards'>"]
    for big, small in ((f"${money(placed)}", "matched to invoices"), (f"${money(held)}", "waiting for a person"),
                       (f"{resolved}/{len(payments)}", "senders identified"),
                       (f"{statuses.count(PaidStatus.PAID)}/{len(statuses)}", "invoices fully paid"),
                       (str(len(needs)), "items to review")):
        out.append(f"<div class='card'><b>{big}</b><span>{small}</span></div>")
    out.append("</div>")

    out.append("<h2>Needs a person</h2>")
    if not needs:
        out.append("<p class='mute'>Nothing is waiting.</p>")
    for u in sorted(needs, key=lambda u: (pay[u.payment_id].timestamp, u.payment_id)):
        p = pay[u.payment_id]
        cand = ""
        if u.candidates:
            cand = " · likely: " + ", ".join(f"{escape(name.get(c.owed_by, c.owed_by))} ({c.person_confidence:.0%})" for c in u.candidates)
        out.append(f"<div class='q'><b>{escape(p.id)}</b> · {p.timestamp:%Y-%m-%d} · “{escape(p.sender_name)}” · "
                   f"${money(p.amount_cents)}{' · memo: ' + escape(repr(p.memo)) if p.memo else ''}<br>"
                   f"<small>${money(u.amount_cents)} held: {escape(u.reason)} <i>({_kind(u.reason)})</i>{cand}</small></div>")

    out.append("<h2>Invoices</h2><div class='wrap'><table><tr><th>Invoice</th><th>Event</th><th>Total</th><th>Status</th><th>Who has paid what</th></tr>")
    for s in sorted(result.invoices, key=lambda s: (inv[s.invoice_id].date, s.invoice_id)):
        i = inv[s.invoice_id]
        shares = "".join(
            f"<div>{escape(name.get(sh.person_id, sh.person_id))}: {money(sh.paid_cents)} of {money(sh.owed_cents)} {_tag(sh.status)}"
            f"<div class='bar'><i style='width:{min(100, 100 * sh.paid_cents // max(sh.owed_cents, 1))}%'></i></div></div>"
            for sh in s.shares)
        note = f"<div class='mute'>{escape('; '.join(s.problems))}</div>" if s.problems else ""
        out.append(f"<tr><td><b>{escape(i.id)}</b><br><span class='mute'>{escape(i.vendor)} · {escape(i.description)}</span>{note}</td>"
                   f"<td>{escape(i.event_id)}<br><span class='mute'>{i.date}</span></td><td class='num'>{money(i.total_cents)}</td>"
                   f"<td>{_tag(s.status)}</td><td>{shares}</td></tr>")
    out.append("</table></div>")

    out.append("<h2>Payments</h2><div class='wrap'><table><tr><th>Payment</th><th>Sender</th><th>Identified as</th><th class='num'>Amount</th><th>Applied to</th></tr>")
    for p in sorted(payments, key=lambda p: (p.timestamp, p.id)):
        i = ident[p.id]
        who = (f"{escape(name.get(i.resolved, i.resolved))} <span class='mute'>{i.confidence:.0%} · {i.method}</span>"
               if i.resolved else "<span class='mute'>not identified</span>")
        ls = links_by.get(p.id, [])
        applied = "".join(
            f"<div>{escape(l.invoice_id)} · {escape(name.get(l.owed_by, l.owed_by))} · {money(l.amount_cents)} "
            f"<span class='mute'>{escape(l.how)}, invoice {l.invoice_confidence:.0%}</span></div>" for l in ls)
        u = held_by.get(p.id)
        if u:
            applied += f"<div class='mute'>held {money(u.amount_cents)}: {escape(u.reason)}</div>"
        out.append(f"<tr><td><b>{escape(p.id)}</b><br><span class='mute'>{p.timestamp:%Y-%m-%d}</span></td>"
                   f"<td>{escape(p.sender_name)}{'<br><span class=mute>' + escape(p.memo) + '</span>' if p.memo else ''}</td>"
                   f"<td>{who}</td><td class='num'>{money(p.amount_cents)}</td><td>{applied}</td></tr>")
    out.append("</table></div>")

    out.append("<h2>Balances</h2><div class='wrap'><table><tr><th>Person</th><th>Event</th><th class='num'>Owed</th><th class='num'>Paid</th><th class='num'>Still open</th></tr>")
    for b in sorted(result.balances, key=lambda b: (b.event_id, b.person_id)):
        open_c = b.owed_cents - b.paid_cents
        out.append(f"<tr><td>{escape(name.get(b.person_id, b.person_id))}</td><td>{escape(b.event_id)}</td>"
                   f"<td class='num'>{money(b.owed_cents)}</td><td class='num'>{money(b.paid_cents)}</td>"
                   f"<td class='num'>{money(open_c) if open_c else '—'}</td></tr>")
    out.append("</table></div></main></html>")
    return "".join(out)


def write_outputs(out_dir: Path, title: str, result, detail, events, invoices, payments, directory, model=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.html").write_text(
        render_html(title, result, detail, events, invoices, payments, directory, model), encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps({
        "links": [asdict(l) for l in result.links],
        "held": [asdict(u) for u in result.unmatched],
        "invoices": [{**asdict(i), "status": str(i.status), "shares": [{**asdict(s), "status": str(s.status)} for s in i.shares]}
                     for i in result.invoices],
        "balances": [asdict(b) for b in result.balances],
        "identities": {k: {"resolved": v.resolved, "confidence": round(v.confidence, 4), "method": v.method,
                           "candidates": {p: round(c, 4) for p, c in v.candidates.items()}, "notes": v.notes}
                       for k, v in detail.identities.items()},
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "links.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["payment", "sender_id", "invoice", "owed_by", "amount", "person_confidence", "invoice_confidence", "how"])
        for l in result.links:
            w.writerow([l.payment_id, l.payer_id, l.invoice_id, l.owed_by, f"{l.amount_cents / 100:.2f}",
                        f"{l.person_confidence:.3f}", f"{l.invoice_confidence:.3f}", l.how])
    with (out_dir / "held.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["payment", "sender", "held_amount", "reason", "kind"])
        by = {p.id: p for p in payments}
        for u in result.unmatched:
            if u.amount_cents >= ROUNDING_CENTS:
                w.writerow([u.payment_id, by[u.payment_id].sender_name, f"{u.amount_cents / 100:.2f}", u.reason, _kind(u.reason)])
    return out_dir / "report.html"
