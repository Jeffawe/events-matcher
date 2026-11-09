"""Data structures for the expense matcher. No matching logic lives here.

Money is stored as integer cents, so splitting never hits float errors.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


class PaidStatus(StrEnum):
    UNPAID = "unpaid"
    PARTIAL = "partial"
    PAID = "paid"


# ---------- What goes in ----------


@dataclass
class Person:
    id: str
    name: str


@dataclass
class Event:
    id: str
    name: str
    start_date: date
    end_date: date
    people: list[Person]  # only the people involved in this event


@dataclass
class Share:
    """Who pays what on an invoice. Exactly one of percent / amount_cents is set."""

    person_id: str
    percent: Decimal | None = None
    amount_cents: int | None = None


@dataclass
class Invoice:
    id: str
    event_id: str
    date: date
    vendor: str
    description: str
    total_cents: int
    shares: list[Share]
    note: str = ""


@dataclass
class Payment:
    id: str
    sender_name: str  # the name the bank shows. Always present, but may match no one
    amount_cents: int
    timestamp: datetime
    memo: str = ""  # whatever the sender typed. Often empty


# ---------- What comes out ----------


@dataclass
class PaymentLink:
    """One piece of a payment applied to one person's share of one invoice."""

    payment_id: str
    payer_id: str  # who sent the money
    invoice_id: str
    owed_by: str  # whose share it pays; differs from payer_id when covering someone
    amount_cents: int
    # Two separate certainties. A later level may move a link to a different invoice for
    # the same person (invoice_confidence < 1) but should never move it to a different
    # person unless person_confidence < 1.
    person_confidence: float = 1.0  # sure this money belongs to owed_by's obligations
    invoice_confidence: float = 1.0  # sure it belongs to THIS invoice rather than another of theirs
    how: str = ""  # which rule produced it: "exact", "oldest first", ...
    basis: str = ""  # name of the learned parameter behind invoice_confidence (see algorithms/params.py)


@dataclass
class UnmatchedAmount:
    """Money that couldn't be linked, held for a person to review."""

    payment_id: str
    amount_cents: int
    reason: str
    payer_id: str | None = None
    # Links that might be right, for the reviewer. Amounts are what could apply.
    candidates: list[PaymentLink] = field(default_factory=list)


@dataclass
class ShareStatus:
    person_id: str
    owed_cents: int
    paid_cents: int
    status: PaidStatus


@dataclass
class InvoiceStatus:
    invoice_id: str
    status: PaidStatus
    shares: list[ShareStatus]
    unassigned_cents: int = 0  # the part no one was given, e.g. a split adding to 90%
    problems: list[str] = field(default_factory=list)  # oddities in the invoice's shares, for the reviewer


@dataclass
class PersonBalance:
    person_id: str
    event_id: str
    owed_cents: int
    paid_cents: int  # paid toward this person's shares, by anyone


@dataclass
class MatchResult:
    links: list[PaymentLink]
    unmatched: list[UnmatchedAmount]
    invoices: list[InvoiceStatus]
    balances: list[PersonBalance]
