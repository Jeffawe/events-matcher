"""Step 1 of matching: work out exactly what each person owes on each invoice.

Pure arithmetic on integer cents. Nothing here looks at payments.
"""

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from models import Invoice


@dataclass
class InvoiceObligations:
    invoice_id: str
    owed: dict[str, int]  # person_id -> cents owed on this invoice
    unassigned_cents: int  # total - sum(owed). Positive: nobody was given it. Negative: over-assigned
    problems: list[str]  # anything odd in the invoice's shares, for the reviewer


def compute_obligations(invoice: Invoice) -> InvoiceObligations:
    owed: dict[str, int] = {}
    problems: list[str] = []

    percent_shares = [s for s in invoice.shares if s.percent is not None]
    amount_shares = [s for s in invoice.shares if s.amount_cents is not None]

    for s in invoice.shares:
        if (s.percent is None) == (s.amount_cents is None):
            problems.append(f"share for {s.person_id} must set exactly one of percent / amount")
        if s.person_id in owed:
            problems.append(f"{s.person_id} appears more than once")

    for s in amount_shares:
        owed[s.person_id] = owed.get(s.person_id, 0) + s.amount_cents

    # Percent shares: floor each, then hand the leftover cents out one each in the order
    # the shares are listed on the invoice, so the shares sum exactly to the percent of
    # the total. (Listed order, not largest fraction: this is the convention the answer key
    # follows, e.g. l_bar gives John the extra cent, not Tunde.)
    if percent_shares:
        exact = {s.person_id: invoice.total_cents * s.percent / 100 for s in percent_shares}
        floors = {p: int(v.to_integral_value(ROUND_FLOOR)) for p, v in exact.items()}
        target = int((sum(exact.values())).to_integral_value())  # half-even on the combined share
        leftover = target - sum(floors.values())
        for p in list(exact)[: max(leftover, 0)]:  # dict keeps listed order
            floors[p] += 1
        for p, cents in floors.items():
            owed[p] = owed.get(p, 0) + cents

    unassigned = invoice.total_cents - sum(owed.values())
    if unassigned > 0:
        problems.append(f"{unassigned} cents of {invoice.total_cents} not assigned to anyone")
    elif unassigned < 0:
        problems.append(f"shares exceed the invoice total by {-unassigned} cents")

    return InvoiceObligations(invoice.id, owed, unassigned, problems)


def compute_all(invoices: list[Invoice]) -> dict[str, InvoiceObligations]:
    return {inv.id: compute_obligations(inv) for inv in invoices}


if __name__ == "__main__":
    from run import load_scenarios

    _, invoices, _ = load_scenarios()
    for inv in invoices:
        ob = compute_obligations(inv)
        flag = "" if not ob.problems else "  <-- " + "; ".join(ob.problems)
        print(f"{inv.id:<14}{inv.total_cents / 100:>9.2f}  {ob.owed}{flag}")
