"""Identity levels, memo plans, one allocation pass, then reconciliation.

    level 1 (exact name) -> level 2 (variants, optional small LLM)
        -> level 3a (memo-explained covering, verified by arithmetic)
        -> allocation (one time-ordered pass)
        -> level 3b (surplus / deficit view and suggestions)

Level 2 conclusions are proposed to the dictionary as "proposed" facts for a human to
approve. Nothing in this run reads those proposals back.
"""

from dataclasses import dataclass

from models import Event, Invoice, Payment

from .allocation import AllocationResult, allocate
from .dictionary import Dictionary, Source
from .identity import Directory, Identity
from .level1_direct import identify_level1
from .level2_variants import identify_level2
from .level3_memo import Plan, Split
from .reconcile import Reconciliation, reconcile
from .reviews import Review


@dataclass
class PipelineResult:
    identities: dict[str, Identity]
    plans: dict[str, Plan]
    allocation: AllocationResult
    reconciliation: Reconciliation
    proposals: int


def run_pipeline(
    events: list[Event],
    invoices: list[Invoice],
    payments: list[Payment],
    dictionary: Dictionary | None = None,
    llm=None,
    reviews: dict[str, Review] | None = None,
) -> PipelineResult:
    dictionary = dictionary or Dictionary()
    reviews = reviews or {}
    directory = Directory.build(events)

    identities = identify_level1(payments, directory, dictionary)
    todo = [p for p in payments if identities[p.id].resolved is None]
    identities.update(identify_level2(todo, directory, invoices, dictionary, llm))

    proposals = 0
    by_id = {p.id: p for p in payments}
    for ident in identities.values():
        if ident.method not in ("variant", "variant+llm"):
            continue
        pay = by_id[ident.payment_id]
        if dictionary.find("alias", pay.sender_name, ident.resolved):
            continue
        source = (Source("model", llm.model) if ident.method == "variant+llm" and llm
                  else Source("rule", "level2"))
        dictionary.propose(
            "alias", pay.sender_name, ident.resolved, source,
            f"{ident.confidence:.0%}: " + "; ".join(ident.notes), [pay.id],
        )
        proposals += 1

    # what a person decided beats anything the pipeline worked out
    for pid, rv in reviews.items():
        if pid not in identities:
            continue
        if rv.payer == "":
            identities[pid] = Identity(pid, {}, None, "human", ["reviewer: nobody on the list"])
        elif rv.payer:
            identities[pid] = Identity(pid, {rv.payer: 1.0}, rv.payer, "human", ["decided in review"])

    plans: dict[str, Plan] = {}
    for pid, rv in reviews.items():
        if rv.links and pid in identities and (identities[pid].resolved or rv.payer == ""):
            plans[pid] = Plan(pid, [Split(l["owed_by"], l["invoice"], int(l["cents"])) for l in rv.links],
                              1.0, "human", "human", ["decided in review"])
    ignored = {pid for pid, rv in reviews.items() if rv.ignore}
    accepted = {pid for pid, rv in reviews.items() if rv.accept}
    alloc = allocate(invoices, payments, identities, directory, events, plans, llm, ignored, accepted)
    plans.update(alloc.plans)  # memo-explained payments are found during allocation
    return PipelineResult(
        identities, plans, alloc, reconcile(invoices, payments, alloc, directory, events), proposals
    )
