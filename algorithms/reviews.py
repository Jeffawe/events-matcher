"""What a person decided about a payment, kept so the next run honours it.

A decision can say any of:
  payer    who really sent it (a person id), or "" for "nobody on the list"
  links    exactly which shares it pays: [{"invoice", "owed_by", "cents"}, ...]
  ignore   not a real receipt (duplicate bank alert, outgoing payment, refund)
  accept   keep the leftover as a credit on the sender (an overpayment); nothing more to do
Later decisions about the same payment replace earlier ones. The log is append-only, so
nothing a reviewer did is ever lost, and decisions double as labels for learning.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import state


@dataclass
class Review:
    payment_id: str
    payer: str | None = None  # None = no decision about the sender; "" = nobody on the list
    links: list[dict] | None = None
    ignore: bool = False
    accept: bool = False  # the money left over is a known credit (an overpayment); stop asking
    note: str = ""
    by: str = ""
    at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ReviewStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else state.reviews_path()

    def add(self, review: Review) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(review), ensure_ascii=False) + "\n")

    def load(self) -> dict[str, Review]:
        out: dict[str, Review] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = Review(**json.loads(line))
                    out[r.payment_id] = r  # last decision wins
        return out
