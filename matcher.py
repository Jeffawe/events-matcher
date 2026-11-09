"""The public entry point.

    result = match_payments(events, invoices, payments)            # models.MatchResult
    result, detail = match_detailed(events, invoices, payments)    # plus the pipeline's working

MatchResult is what the data model promised: the links, the money that could not be placed,
the status of every invoice and share, and each person's balance per event. `detail` adds who
the pipeline thinks sent each payment, why, and the reviewer-facing suggestions.
"""

from models import (
    Event, Invoice, InvoiceStatus, MatchResult, PaidStatus, Payment, PaymentLink,
    PersonBalance, ShareStatus, UnmatchedAmount,
)
from obligations import compute_obligations

from algorithms.dictionary import Dictionary
from algorithms.pipeline import PipelineResult, run_pipeline
from algorithms.level2_variants import obligation_index


def _status(owed: int, paid: int) -> PaidStatus:
    if paid >= owed and owed > 0:
        return PaidStatus.PAID
    return PaidStatus.PARTIAL if paid > 0 else PaidStatus.UNPAID


def invoice_statuses(invoices: list[Invoice], links: list[PaymentLink]) -> list[InvoiceStatus]:
    paid: dict[tuple[str, str], int] = {}
    for l in links:
        paid[(l.invoice_id, l.owed_by)] = paid.get((l.invoice_id, l.owed_by), 0) + l.amount_cents
    out = []
    for inv in invoices:
        ob = compute_obligations(inv)
        shares = [
            ShareStatus(p, owed, paid.get((inv.id, p), 0), _status(owed, paid.get((inv.id, p), 0)))
            for p, owed in ob.owed.items()
        ]
        total_paid = sum(s.paid_cents for s in shares)
        # an invoice is only PAID when every share is paid AND nobody is left unassigned
        if shares and all(s.status == PaidStatus.PAID for s in shares) and ob.unassigned_cents == 0:
            status = PaidStatus.PAID
        else:
            status = PaidStatus.PARTIAL if total_paid > 0 else PaidStatus.UNPAID
        out.append(InvoiceStatus(inv.id, status, shares, max(ob.unassigned_cents, 0), ob.problems))
    return out


def _candidate_links(unmatched: list[UnmatchedAmount], detail: PipelineResult, payments, invoices):
    """For payments whose sender is unknown, show the reviewer where it would go for each likely person."""
    owed = obligation_index(invoices)
    by_id = {p.id: p for p in payments}
    inv_of = {}
    for inv in invoices:
        for pid, c in compute_obligations(inv).owed.items():
            if c > 0:
                inv_of.setdefault(pid, []).append((inv.date, inv.id, c))
    for u in unmatched:
        ident = detail.identities.get(u.payment_id)
        if not ident or ident.resolved or u.payer_id:
            continue
        pay = by_id[u.payment_id]
        for pid, prob in sorted(ident.candidates.items(), key=lambda x: -x[1]):
            if prob < 0.1:
                continue
            rows = sorted(r for r in inv_of.get(pid, []) if r[0] <= pay.timestamp.date())
            if rows:
                _, inv_id, cents = next((r for r in rows if r[2] == pay.amount_cents), rows[0])
                u.candidates.append(PaymentLink(pay.id, pid, inv_id, pid, min(cents, pay.amount_cents),
                                                prob, 0.5, "candidate"))


def match_detailed(
    events: list[Event],
    invoices: list[Invoice],
    payments: list[Payment],
    dictionary: Dictionary | None = None,
    llm=None,
    reviews: dict | None = None,
) -> tuple[MatchResult, PipelineResult]:
    detail = run_pipeline(events, invoices, payments, dictionary, llm, reviews)
    alloc = detail.allocation
    _candidate_links(alloc.unmatched, detail, payments, invoices)
    balances = [PersonBalance(b.person_id, b.event_id, b.owed_cents, b.paid_cents)
                for b in detail.reconciliation.balances]
    return MatchResult(alloc.links, alloc.unmatched, invoice_statuses(invoices, alloc.links), balances), detail


def match_payments(events, invoices, payments, dictionary=None, llm=None, reviews=None) -> MatchResult:
    """Link every payment to the invoices it pays, and report what's still owed."""
    return match_detailed(events, invoices, payments, dictionary, llm, reviews)[0]
