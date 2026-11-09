"""Every tunable probability and confidence in one place, and able to LEARN.

Each parameter is a prior (the best guess before seeing data) plus a strength (how many
observations that guess is worth). With no data the value is the prior, so behaviour is
exactly what the code was designed with. Given labelled outcomes it moves toward the
observed frequency:

    value = (strength * prior + hits) / (strength + trials)

A prior worth 20 observations is overridden once there are a few hundred; a handful of
labels barely moves it. Values are kept away from 0 and 1 so one lucky run can't make
anything certain.

What is NOT learned, on purpose:
  * RESOLVE_P in level2_variants - how sure to be before acting is a risk decision, not a
    frequency. learn.py prints what precision each probability band actually achieved.
  * UNKNOWN_PRIOR - the share of senders who aren't on the list needs labels saying
    "this sender is nobody listed", which the answer key does not give.

Where the observations live: algorithms/params.json holds one entry per labelled payment,
so labelling the same payment twice replaces its entry instead of counting it twice.
"""

import json
from contextlib import contextmanager
from pathlib import Path

from . import state

FLOOR = 0.001

# name: (prior value, strength in pseudo-observations)
DEFAULTS: dict[str, tuple[float, int]] = {
    # level 2: P(name match level | this person IS the sender) / P(level | someone else is)
    **{f"name.m.{lvl}": (v, 20) for lvl, v in {5: 0.95, 4: 0.90, 3: 0.85, 2: 0.60, 1: 0.30}.items()},
    **{f"name.eps.{lvl}": (v, 20) for lvl, v in {5: 0.01, 4: 0.02, 3: 0.01, 2: 0.06, 1: 0.08}.items()},
    "name.m.any": (0.90, 20),  # P(sender looks at least a bit like them | they are the sender)
    "name.eps.any": (0.15, 20),  # ... | they are not
    "ctx.m.owes": (0.85, 20),  # the sender owes something dated by then
    "ctx.m.share": (0.50, 20),  # the amount equals one of the sender's shares
    "ctx.m.fits": (0.95, 20),  # the amount fits within what the sender owes
    "llm.m": (0.80, 10),  # the small model names the sender | it is them   (learned per model)
    "llm.u": (0.08, 10),  # ... names a given competing candidate | someone else is the sender
    # confidence that a link sits on the right invoice, by how the link was chosen
    "inv.subset": (0.90, 10),  # the payment equals the exact sum of some whole open shares (unique)
    "inv.exact": (1.0, 10),  # amount equals one share (divided by the number of equal shares)
    "inv.oldest_first": (0.50, 10),
    "inv.event_hint": (0.70, 10),
    "inv.items.amounts": (0.95, 10),  # every chosen share named by an amount in the memo
    "inv.items.words": (0.85, 10),  # chosen by matching words, sum exact
    "inv.items.model": (0.80, 10),  # chosen by the small model, sum exact
    "inv.items.tie": (0.60, 10),  # several equal sets, took the oldest
    "inv.memo_unique": (1.0, 10),  # memo-named people's shares add up on exactly one invoice
    "inv.memo_picked": (0.70, 10),  # several invoices add up; the memo's words chose one
    "person.covered": (0.90, 10),  # a person the memo says the payer covered
}


class Params:
    def __init__(self, counts: dict[str, list[int]] | None = None):
        self.counts = counts or {}  # name -> [hits, trials]

    def get(self, name: str) -> float:
        """A name like "llm.m@qwen2.5:3b" is learned separately per model, starting from the
        prior of "llm.m"."""
        prior, strength = DEFAULTS[name.split("@")[0]]
        hits, trials = self.counts.get(name, (0, 0))
        v = (strength * prior + hits) / (strength + trials)
        return min(max(v, FLOOR), 1 - FLOOR) if prior < 1 else min(max(v, FLOOR), 1.0)

    def add(self, name: str, hit: bool | int | float, trials: int = 1) -> None:
        c = self.counts.setdefault(name, [0, 0])
        c[0] += float(hit)
        c[1] += trials

    def table(self) -> list[tuple[str, float, float, int]]:
        """(name, prior, current value, trials seen)"""
        names = list(DEFAULTS) + [n for n in self.counts if "@" in n]
        return [(n, DEFAULTS[n.split("@")[0]][0], self.get(n), self.counts.get(n, (0, 0))[1]) for n in names]


def aggregate(observations: dict[str, dict]) -> Params:
    """Turn per-payment observations (see learn.py) into parameter counts."""
    p = Params()
    for obs in observations.values():
        ident = obs.get("identity")
        if ident:
            s, n_other = ident["sender"], ident["n_other"]
            for lvl in range(1, 6):
                p.add(f"name.m.{lvl}", s["level"] == lvl)
                p.add(f"name.eps.{lvl}", ident["other_levels"].get(str(lvl), 0), n_other)
            p.add("name.m.any", s["level"] >= 1)
            p.add("name.eps.any", sum(v for k, v in ident["other_levels"].items() if int(k) >= 1), n_other)
            for flag in ("owes", "share", "fits"):
                if s[flag] is not None:  # None: the evidence did not apply to this payment
                    p.add(f"ctx.m.{flag}", s[flag])
            if "llm" in ident:
                m = ident["llm"]
                p.add(f"llm.m@{m['model']}", m["sender_voted"])
                if m["n_competing"]:  # u: chance of naming a given COMPETING candidate who isn't the sender
                    p.add(f"llm.u@{m['model']}", m["competing_voted"], m["n_competing"])
        amb = obs.get("ambiguous")
        if amb:  # the sender is one of k plausible people and nobody can tell which
            k = amb["n_candidates"]
            if amb["voted_candidate"]:  # a vote for one of them is a coin flip, so (k-1)/k of it is a wrong vote
                p.add(f"llm.u@{amb['model']}", (k - 1) / k, k - 1)
            else:
                p.add(f"llm.u@{amb['model']}", 0, k - 1)
        for link in obs.get("links", []):
            p.add(link["basis"], link["hit"])
    return p


ACTIVE = Params()


def load(path: Path | None = None) -> Params:
    path = path or state.params_path()
    if path.exists():
        return aggregate(json.loads(path.read_text(encoding="utf-8")).get("observations", {}))
    return Params()


def get(name: str) -> float:
    return ACTIVE.get(name)


@contextmanager
def use(params: Params):
    """Temporarily run the pipeline with different parameters (cross-validation)."""
    global ACTIVE
    saved, ACTIVE = ACTIVE, params
    try:
        yield
    finally:
        ACTIVE = saved


ACTIVE = load()
