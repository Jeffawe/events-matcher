"""Where the system keeps what it has learned and what people have decided.

    state/dictionary.jsonl   name aliases (proposed by the pipeline, approved by people)
    state/reviews.jsonl      every decision a reviewer made about a payment (append-only)
    state/params.json        labelled observations the tunable numbers are learned from

Override the folder with the MATCH_STATE environment variable (for a demo, point it at a
fresh folder so nothing carries over).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def state_dir() -> Path:
    d = Path(os.environ.get("MATCH_STATE", ROOT / "state"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def dictionary_path() -> Path:
    return state_dir() / "dictionary.jsonl"


def reviews_path() -> Path:
    return state_dir() / "reviews.jsonl"


def params_path() -> Path:
    return state_dir() / "params.json"
