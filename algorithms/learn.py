"""Learn the pipeline's tunable numbers from labelled payments.

    python3 -m algorithms.learn --cv       # honest test: learn on half, score the other half
    python3 -m algorithms.learn --report   # what every parameter would become, and from how much data
    python3 -m algorithms.learn --save     # store the observations in algorithms/params.json

A label is: who really sent the payment, and which (invoice, person, amount) links it should
have. Today the labels come from scenarios/expected.json. In use they would come from the
human review queue (a reviewer confirms or corrects a payment). The format is the same:
a dict {payment_id: Label}. See params.py for how observations become parameters.
"""

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms import params as P  # noqa: E402
from algorithms import state  # noqa: E402
from algorithms.identity import Directory  # noqa: E402
from algorithms.level2_variants import memo_names_a_person, obligation_index, pair_features  # noqa: E402
from algorithms.pipeline import run_pipeline  # noqa: E402
from run import DATA, load_scenarios, to_cents  # noqa: E402


@dataclass
class Label:
    payer: str | None  # who really sent it; None when the key says it is unknown / held
    links: set  # {(invoice_id, owed_by, cents)}; empty when the key did not place it
    candidates: frozenset = frozenset()  # when payer is None: the people it could be (ambiguous)


def load_labels() -> dict[str, Label]:
    rows = json.loads((DATA / "expected.json").read_text())["payments"]
    return {
        r["payment_id"]: Label(
            r["payer"],
            {(l["invoice"], l["owed_by"], to_cents(l["amount"])) for l in r["links"]},
            frozenset(c["owed_by"] for c in r.get("candidates", [])),
        )
        for r in rows
    }


def labels_from_reviews() -> dict[str, Label]:
    """Reviewers' decisions are labels too: who sent it, and which shares it paid."""
    from algorithms.reviews import ReviewStore
    out = {}
    for pid, rv in ReviewStore().load().items():
        if rv.ignore:
            continue
        out[pid] = Label(rv.payer or None, {(l["invoice"], l["owed_by"], int(l["cents"])) for l in (rv.links or [])})
    return out


def observe(res, events, invoices, payments, labels) -> dict[str, dict]:
    """One observation per labelled payment (replaces any earlier one for the same payment)."""
    directory, owed = Directory.build(events), obligation_index(invoices)
    by_id = {p.id: p for p in payments}
    obs: dict[str, dict] = {}
    for pid, lab in labels.items():
        o: dict = {}
        ident = res.identities[pid]
        if lab.payer and ident.method != "exact":  # the payment went through level 2
            levels, has = pair_features(by_id[pid], directory, owed)
            if lab.payer in levels:
                others = [p for p in levels if p != lab.payer]
                entry = {
                    "sender": {"level": levels[lab.payer], "owes": has["owes"][lab.payer],
                               "share": has["share"][lab.payer],
                               "fits": None if memo_names_a_person(by_id[pid], directory) else has["fits"][lab.payer]},
                    "other_levels": dict(Counter(str(levels[p]) for p in others if levels[p] >= 1)),
                    "n_other": len(others),
                }
                if ident.llm_vote is not False:
                    competing = [p for p in others if levels[p] >= 1]  # people the name could also be
                    entry["llm"] = {"model": ident.llm_model, "sender_voted": ident.llm_vote == lab.payer,
                                    "competing_voted": int(ident.llm_vote in competing),
                                    "n_competing": len(competing)}
                o["identity"] = entry
        elif lab.payer is None and len(lab.candidates) >= 2 and ident.llm_vote is not False:
            # the key says it is one of several people and nobody can tell which: a model that
            # names one of them is guessing
            o["ambiguous"] = {"model": ident.llm_model, "n_candidates": len(lab.candidates),
                              "voted_candidate": ident.llm_vote in lab.candidates}
        if lab.links:
            owners = {l[1] for l in lab.links}
            links = []
            for l in res.allocation.links:
                if l.payment_id != pid or not l.basis:
                    continue
                links.append({"basis": l.basis, "hit": (l.invoice_id, l.owed_by, l.amount_cents) in lab.links})
                if l.owed_by != l.payer_id:
                    links.append({"basis": "person.covered", "hit": l.owed_by in owners})
            o["links"] = links
        if o:
            obs[pid] = o
    return obs


def evaluate(res, labels, ids) -> dict[str, float]:
    right = wrong = unsure = 0
    brier_id, n_id = 0.0, 0
    brier_inv, n_inv = 0.0, 0
    for pid in ids:
        lab = labels[pid]
        ident = res.identities[pid]
        if lab.payer:
            if ident.resolved is None:
                unsure += 1
            elif ident.resolved == lab.payer:
                right += 1
            else:
                wrong += 1
            if ident.method != "exact" and ident.candidates:
                top = max(ident.candidates, key=ident.candidates.get)
                brier_id += (ident.candidates[top] - (top == lab.payer)) ** 2
                n_id += 1
        if lab.links:
            for l in res.allocation.links:
                if l.payment_id == pid:
                    brier_inv += (l.invoice_confidence - ((l.invoice_id, l.owed_by, l.amount_cents) in lab.links)) ** 2
                    n_inv += 1
    return {"right": right, "wrong": wrong, "uncertain": unsure,
            "brier_identity": brier_id / max(n_id, 1), "brier_invoice": brier_inv / max(n_inv, 1)}


def cross_validate(events, invoices, payments, labels, reps=5, llm=None) -> None:
    with P.use(P.Params()):  # observations are taken under the hand-set priors
        base = run_pipeline(events, invoices, payments, llm=llm)
    obs = observe(base, events, invoices, payments, labels)
    ids = sorted(labels)
    hand, learned = Counter(), Counter()
    runs = 0
    for seed in range(reps):
        order = ids[:]
        random.Random(seed).shuffle(order)
        folds = [order[0::2], order[1::2]]
        for k in range(2):
            test, train = folds[k], folds[1 - k]
            params = P.aggregate({i: obs[i] for i in train if i in obs})
            for name, prm in (("hand", P.Params()), ("learned", params)):
                with P.use(prm):
                    res = run_pipeline(events, invoices, payments, llm=llm)
                (hand if name == "hand" else learned).update(evaluate(res, labels, test))
            runs += 1
    print(f"CROSS-VALIDATION: learn on half of the {len(ids)} labelled payments, score the other half "
          f"({reps} random splits x 2 folds, averaged)")
    print(f"  {'':<34}{'hand-set':>10}{'learned':>10}")
    for label, key, fmt in (("senders resolved right", "right", "{:.1f}"), ("senders resolved WRONG", "wrong", "{:.2f}"),
                            ("senders left uncertain", "uncertain", "{:.1f}"),
                            ("identity probability, Brier", "brier_identity", "{:.4f}"),
                            ("invoice confidence, Brier", "brier_invoice", "{:.4f}")):
        print(f"  {label:<34}{fmt.format(hand[key] / runs):>10}{fmt.format(learned[key] / runs):>10}")
    print("  (Brier: mean squared gap between stated confidence and what happened; lower is better.)")


def report(obs) -> None:
    prm = P.aggregate(obs)
    print(f"{'parameter':<20}{'hand-set':>10}{'learned':>10}{'trials':>9}")
    for name, prior, value, trials in prm.table():
        if trials:
            print(f"{name:<20}{prior:>10.3f}{value:>10.3f}{trials:>9}{'   <-- moved' if abs(value - prior) > 0.1 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--reviews", action="store_true", help="learn from reviewers' decisions (state/reviews.jsonl) instead of the answer key")
    ap.add_argument("--llm", action="store_true", help="include the Ollama model (OLLAMA_MODEL); cached answers need no memory")
    args = ap.parse_args()
    from algorithms.llm_ollama import OllamaVoter
    llm = OllamaVoter() if args.llm else None
    events, invoices, payments = load_scenarios()
    labels = labels_from_reviews() if args.reviews else load_labels()
    if args.cv:
        cross_validate(events, invoices, payments, labels, llm=llm)
    if args.report or args.save:
        with P.use(P.Params()):
            res = run_pipeline(events, invoices, payments, llm=llm)
        obs = observe(res, events, invoices, payments, labels)
        if args.report:
            report(obs)
        if args.save:
            path = state.params_path()
            old = json.loads(path.read_text())["observations"] if path.exists() else {}
            old.update(obs)  # same payment labelled again replaces its entry
            path.write_text(json.dumps({"observations": old}, indent=1))
            print(f"saved {len(old)} labelled payments to {path}")
    if not (args.cv or args.report or args.save):
        ap.print_help()


if __name__ == "__main__":
    main()
