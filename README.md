# Payment matcher

Matches incoming payments to the invoices they pay when the data is messy: names are abbreviated or
misspelled, one payment covers several invoices or several people, bank alerts duplicate payments,
and memos are free text. Every answer carries a confidence, and anything the system cannot be sure
of is held for a person instead of guessed. People's decisions are remembered and the system learns
from them.

Python 3.11+, standard library only. An optional local model (Ollama) helps with the hardest names.

**Try it in your browser: https://jeffawe.github.io/events-matcher/** — runs this exact Python code in the page (nothing is uploaded) on the sample data, a freshly generated messy world, or your own files.

## Quick start

```bash
python3 matchpay.py doctor           # is this machine ready? (checks Python, data, Ollama)
python3 matchpay.py demo --open      # fresh run on the sample data; opens out/report.html
python3 matchpay.py review           # work through what the system could not settle
python3 matchpay.py run --open       # apply your decisions and rebuild the report
```

Everything else:

| Command | What it does |
|---|---|
| `run` | Match payments to invoices. Writes `out/report.html`, `result.json`, `links.csv`, `held.csv`. |
| `demo` | Same, from a clean slate (no earlier decisions or learned aliases), so it always looks the same. |
| `review` | Decide held payments one at a time. `--aliases` approves or rejects name aliases the system proposed. `--list` shows the queue. `--apply FILE` takes decisions from JSON. |
| `doctor` | Checks the machine and says what to fix. |
| `pull-model` | Downloads the recommended local model (`--small` for the lighter one). |
| `generate` | Makes a brand-new messy scenario with known answers (`--seed N`). |
| `score` / `stress` / `learn` | Evaluation tools. They take their own options (`score --data DIR`, `learn --cv`). |

Add `--llm` to `run`, `demo` or `review` to use the local model. `--data DIR` points at a different
scenario folder; `--state DIR` keeps decisions somewhere else.

In code: `from matcher import match_payments, match_detailed`. `match_payments(events, invoices,
payments)` returns a `models.MatchResult`: the links, the money not placed, every invoice's and share's
status, and each person's balance per event.

## Data format

A scenario folder holds one JSON file per event, plus the payments:

```
scenarios/events/lisbon.json   {"id", "name", "start_date", "end_date",
                                "people": [{"id", "name"}],
                                "invoices": [{"id", "date", "vendor", "description", "total": "2000.00",
                                              "shares": [{"person": "john", "percent": "30"}   # or "amount": "600.00"
                                                         ...]}]}
scenarios/payments.json        [{"id", "sender_name", "amount": "300.00", "timestamp": "2026-03-03T09:12:00", "memo": ""}]
scenarios/expected.json        optional answer key, used only by `score` (see the sample)
```

Person ids are shared across events. Percent shares round down and leftover cents go to the shares in listed order.

## How it works

Identity (who sent it) is decided first for every payment. Allocation (which invoice) then runs once,
over all payments in time order, so each payment sees what earlier ones already paid.

| Step | File | What it does |
|---|---|---|
| Obligations | `obligations.py` | What each person owes on each invoice, in integer cents. Leftover cents go to shares in listed order. |
| Level 1 | `algorithms/level1_direct.py` | Sender name equals exactly one person's name, or a trusted alias. |
| Level 2 | `algorithms/level2_variants.py` | Name variants (initials, typos, nicknames, swapped order, extra words) scored with a Fellegi-Sunter model plus context. Probabilities are split between look-alikes; only a clear winner is acted on. |
| Local model | `algorithms/llm_ollama.py` | Votes on hard names. It may confirm a candidate the rules already favour strongly; it never chooses between close ones. |
| Level 3a | `algorithms/level3_memo.py` | "Covering Amara's share": what the named people still owe on one invoice must add up to the payment exactly. A first name that fits two people is tried as each. |
| Allocation | `algorithms/allocation.py` | Puts money on the sender's own shares. Never exceeds what is owed; surplus stays unplaced. Holds bank alerts, duplicates, outgoing payments and refunds. |
| Level 4 | `algorithms/level4_items.py` | The memo's amounts and words choose which invoices ("cab 92 + cabin 360", "less the $48 Mina covered"). A payment that equals the exact sum of some shares (with real cents) is matched without any memo. |
| Level 3b | `algorithms/reconcile.py` | Who still owes what; ranked suggestions for unplaced money. Never applied automatically. |
| Review | `algorithms/review.py` | The human loop. Decisions go to `state/reviews.jsonl` and override the pipeline on the next run. |

## Confidence

Each link has two confidences. `person_confidence` is how sure we are whose obligation the money
settles. `invoice_confidence` is how sure we are of the specific invoice. A later step may move a link
to another invoice of the same person when the second is low; the first should stay put.

## What is kept, and learning

Everything people decide and everything learned lives in `state/` (or `MATCH_STATE`):

- `dictionary.jsonl`: name aliases. The system only *proposes*; a person approves. A human decision is locked.
- `reviews.jsonl`: every review decision, append-only.
- `params.json`: labelled observations the tunable numbers are learned from (`learn --save`, or `learn --reviews --save` to learn from reviewers).

Every tunable probability lives in `algorithms/params.py` as a prior that moves toward observed
frequencies as labelled payments arrive. Nothing is learned until you save labels, so default behaviour
is the designed behaviour. English-specific wording (bank alerts, "you paid") is isolated in
`algorithms/lexicon.py`.

**Learn confidence from your own data.** Name-matching numbers carry over between datasets. Invoice
confidences do not: how often "oldest first" is right depends on how the people in the data pay. In our
tests, numbers learned on synthetic data helped other synthetic data but made the original sample
worse. Treat the shipped confidences as starting guesses and let reviews calibrate them.

## The local model (recommended: `llama3.1:8b`, about 5 GB)

```bash
python3 matchpay.py pull-model       # or: ollama pull llama3.1:8b
python3 matchpay.py run --llm
```

We recommend **llama3.1:8b**. It uses about 5 GB of memory while loaded. Answers are cached in
`algorithms/.cache/llm_votes.jsonl`, so a repeat run needs no model at all, and the model unloads 30
seconds after its last question (`OLLAMA_KEEP_ALIVE`). Change models with `OLLAMA_MODEL`.

Why not the smaller `qwen2.5:3b` (about 2 GB)? On the 15 hard-name questions in the sample data, the 8B
model gave no wrong answers. The 3B model missed one correct answer ("Mara" is Amara) and guessed on three
names that really are ambiguous ("Kris Lee", "A KIM", "S LEE"). A guard now stops a guess from becoming a
resolved identity, so the 3B model is a reasonable choice on a small machine. The pipeline also works with
no model at all: it simply leaves a few more senders for a person to review.

## Limits

- The synthetic worlds were written by the same people who wrote the matcher, so they can share its blind spots. They check generalisation; they do not replace real data.
- Default probabilities are estimates until learned from your own labelled payments.
- Wording rules and the sound-alike name rule assume English. Edit `lexicon.py`, or set `USE_PHONETIC = False`.
- One currency is assumed. Negative or zero amounts are held for review.
- Senders that match nobody ("MOBILE DEPOSIT", unreadable names) are held for review by design.
- Some payments stay ambiguous even with arithmetic, such as p083 ("…train 125") which fits two equally good invoices. They are placed at low confidence or held.

## License

MIT. See [LICENSE](LICENSE).
