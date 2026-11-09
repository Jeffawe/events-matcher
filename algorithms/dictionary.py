"""The learning dictionary: facts about names, with provenance and trust.

Rules for how it is used:
  * Anyone (a level, a model, a human) can PROPOSE a fact. Nobody edits it directly.
  * A fact only influences matching once its status is "trusted".
  * A human decision is locked: no rule or model can overturn it.
  * Storage is an append-only JSONL log. The current state is replayed from it, so
    a bad source can be audited and reverted by deciding its facts "rejected".

Fact kinds:
  alias     normalised sender name  ->  person_id      ("john a" is john)
  negative  normalised sender name  -/-> person_id     ("kris" is NOT chris)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .normalize import normalize

PROPOSED, TRUSTED, REJECTED = "proposed", "trusted", "rejected"


@dataclass
class Source:
    kind: str  # "rule" | "human" | "model"
    name: str  # e.g. "level2", "alice", "claude-haiku-5-5"
    version: str = ""


@dataclass
class Fact:
    id: str
    kind: str  # "alias" | "negative"
    key: str  # normalised sender name
    person_id: str
    scope: str | None  # None = global, otherwise an event id
    source: Source
    reason: str
    evidence: list[str] = field(default_factory=list)  # payment ids that support it
    status: str = PROPOSED
    locked: bool = False  # set by a human decision


class Dictionary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.facts: dict[str, Fact] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    # ---------- writing ----------

    def propose(
        self,
        kind: str,
        raw_name: str,
        person_id: str,
        source: Source,
        reason: str,
        evidence: list[str] | None = None,
        scope: str | None = None,
    ) -> str:
        assert kind in ("alias", "negative")
        key = normalize(raw_name)
        fact_id = f"f{len(self.facts) + 1:05d}"
        status = TRUSTED if source.kind == "human" else PROPOSED
        fact = Fact(fact_id, kind, key, person_id, scope, source, reason, evidence or [], status,
                    locked=source.kind == "human")
        self._record({"op": "propose", "fact": asdict(fact)})
        return fact_id

    def find(self, kind: str, raw_name: str, person_id: str, scope: str | None = None) -> str | None:
        """Id of an existing, not-rejected fact with this key, so we don't propose twice."""
        key = normalize(raw_name)
        for f in self.facts.values():
            if (f.kind, f.key, f.person_id, f.scope) == (kind, key, person_id, scope) and f.status != REJECTED:
                return f.id
        return None

    def decide(self, fact_id: str, status: str, source: Source, reason: str = "") -> bool:
        """Promote or reject a fact. Returns False if a human lock blocks it."""
        fact = self.facts[fact_id]
        if fact.locked and source.kind != "human":
            return False
        assert status in (TRUSTED, REJECTED)
        self._record({"op": "decide", "fact_id": fact_id, "status": status,
                      "source": asdict(source), "reason": reason})
        return True

    # ---------- reading ----------

    def resolve(self, raw_name: str, scope: str | None = None, include_proposed: bool = False) -> tuple[set[str], set[str]]:
        """Return (people this name is known to be, people it is known NOT to be).

        Only trusted facts count unless include_proposed. Global facts always apply;
        scoped facts only when `scope` matches.
        """
        key = normalize(raw_name)
        allowed = {TRUSTED, PROPOSED} if include_proposed else {TRUSTED}
        is_, is_not = set(), set()
        for f in self.facts.values():
            if f.key != key or f.status not in allowed:
                continue
            if f.scope is not None and f.scope != scope:
                continue
            (is_ if f.kind == "alias" else is_not).add(f.person_id)
        return is_ - is_not, is_not

    # ---------- storage ----------

    def _record(self, event: dict) -> None:
        event["at"] = datetime.now().isoformat(timespec="seconds")
        self._apply(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _apply(self, event: dict) -> None:
        if event["op"] == "propose":
            raw = dict(event["fact"])  # copy: the event itself must stay JSON-serialisable
            raw["source"] = Source(**raw["source"])
            fact = Fact(**raw)
            self.facts[fact.id] = fact
        elif event["op"] == "decide":
            fact = self.facts[event["fact_id"]]
            if fact.locked and event["source"]["kind"] != "human":
                return  # replaying an old log must respect the lock too
            fact.status = event["status"]
            if event["source"]["kind"] == "human":
                fact.locked = True
