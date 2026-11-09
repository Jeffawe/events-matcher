"""Make a brand-new messy scenario with known answers, to test the pipeline on data it was
never tuned on.

    python3 matchpay.py generate --seed 7 --out scenarios_synth/seed7

Truth is built in: the generator decides who pays what, then disguises it. Every payment
records the kind of mess it received ("noise") and the habit it came from ("pattern"), so a
failure can be traced to a cause. Same file layout as scenarios/, so everything that reads
scenarios/ works on it (python3 -m algorithms.score --data scenarios_synth/seed7).

Honest limits: the generator is written by the same people who wrote the matcher, so it can
share their blind spots. Identity results are the fair test (names are disguised in ways the
matcher does not special-case). Invoice results for lump-sum payments partly depend on
whether the generator paid oldest-first, which it does only 60% of the time on purpose.
"""

import json
import random
import sys
import unicodedata
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obligations import compute_obligations  # noqa: E402

FIRST = ("Alexander Amina Ben Carla Dmitri Elena Farid Grace Hiro Imani Jonas Kavya Liam Mateus Nadia Omar "
         "Priya Quinn Rosa Sofia Tariq Uma Victor Wen Xavier Yara Zane Aisha Bruno Chloe Dev Esme Finn Gita "
         "Hana Ivan Jada Kofi Lena Marco Noor Owen Paula Rafael Selin Theo Una Vera Wale Yusuf Zara").split()
LAST = ("Okafor Silva Tanaka Novak Haddad Larsen Moreau Bianchi Costa Petrov Nguyen Rossi Kowalski Mensah "
        "Ibrahim Fischer Santos Khan Murphy Aydin Lindqvist Vargas Osei Duarte Sato Varga Reyes Adler Brandt Cole").split()
NICK = {"alexander": ["alex", "xander"], "benjamin": ["ben"], "elena": ["lena"], "dmitri": ["dima"],
        "rafael": ["rafa"], "theo": ["theodore"], "marco": ["marc"], "jonas": ["jon"], "victor": ["vic"]}
PHONETIC = [("ph", "f"), ("ck", "k"), ("ie", "y"), ("y", "i"), ("sch", "sk"), ("th", "t"), ("ou", "u")]
CJK = {"Wen Tanaka": "文田", "Hana Sato": "花佐藤"}
EVENT_NAMES = ["Porto offsite", "Denver conference", "Seoul retreat", "Nairobi workshop", "Oslo summit",
               "Valencia sprint", "Quebec trip", "Cusco expedition"]
TEMPLATES = [  # vendor, description, typical total range in dollars
    ("Hotel {}", "Lodging for the group", (900, 3200)), ("{} Bistro", "Group dinner", (180, 700)),
    ("City Cabs", "Airport transfers", (60, 260)), ("{} Museum", "Museum tickets", (90, 400)),
    ("Blue Water Tours", "Boat tour", (150, 800)), ("{} Rentals", "Rental van", (200, 900)),
    ("Market Co", "Groceries and supplies", (80, 420)), ("Sing Along", "Karaoke room", (120, 500)),
    ("{} Hall", "Conference passes", (500, 2400)), ("Sparkle Services", "Cleaning fee", (60, 240)),
    ("Summit Gear", "Equipment rental", (100, 600)), ("{} Cafe", "Breakfast", (60, 260))]
PLACES = ["Alma", "Riverside", "Harbor", "Central", "Lantern", "Cedar", "Atlas", "Orchid"]
UNREADABLE = ["MOBILE DEPOSIT", "ACH CREDIT 4821", "BANK TRANSFER", "ZELLE PAYMENT 0097", "INCOMING WIRE"]
STRANGERS = ["Folake Abara", "Marguerite Dubois", "Tomasz Wrona", "Ilse Hartmann"]


def ascii_fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def make_people(rng: random.Random, n: int) -> list[dict]:
    people, used = [], set()

    def add(first, last):
        pid = ascii_fold(first).lower() + "_" + ascii_fold(last).lower()[:1]
        k = 1
        while pid in used:
            k += 1
            pid = f"{pid.rstrip('0123456789')}{k}"
        used.add(pid)
        people.append({"id": pid, "name": f"{first} {last}"})

    while len(people) < n:
        first, last = rng.choice(FIRST), rng.choice(LAST)
        if any(p["name"] == f"{first} {last}" for p in people):
            continue
        add(first, last)
        if rng.random() < 0.30 and len(people) < n:  # a look-alike: same first or same last name
            if rng.random() < 0.5:
                add(first, rng.choice([x for x in LAST if x != last]))
            else:
                add(rng.choice([x for x in FIRST if x != first]), last)
    return people[:n]


def variant(name: str, rng: random.Random, twins: set[str]) -> tuple[str, str]:
    """A disguised version of a name and the kind of disguise."""
    first, *rest = name.split()
    last = rest[-1] if rest else ""
    kind = rng.choices(
        ["exact", "upper", "lower", "first only", "initial+last", "first+initial", "last first", "typo",
         "nickname", "phonetic", "no accents", "extra word", "cjk"],
        [30, 8, 4, 8, 6, 5, 5, 8, 6, 4, 3, 4, 2])[0]
    if kind == "upper":
        return name.upper(), kind
    if kind == "lower":
        return name.lower(), kind
    if kind == "first only":
        return first, kind
    if kind == "initial+last":
        return f"{first[0]}. {last}", kind
    if kind == "first+initial":
        return f"{first} {last[:1]}", kind
    if kind == "last first":
        return f"{last} {first}", kind
    if kind == "typo":
        w = max(name.split(), key=len)
        i = rng.randint(1, max(1, len(w) - 2))
        bad = w[:i] + w[i + 1:] if rng.random() < .5 or len(w) < 4 else w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return name.replace(w, bad, 1), kind
    if kind == "nickname" and first.lower() in NICK:
        return f"{rng.choice(NICK[first.lower()]).title()} {last}", kind
    if kind == "phonetic":
        for a, b in PHONETIC:
            if a in first.lower():
                return f"{first.lower().replace(a, b, 1).title()} {last}", kind
    if kind == "extra word":
        return f"{name} {rng.choice(['Jr', 'Sr', '& Co', '(joint)'])}", kind
    if kind == "cjk" and name in CJK:
        return CJK[name], kind
    if kind == "no accents":
        return ascii_fold(name), kind
    return name, "exact"


def generate(seed: int, out: Path, n_events: int = 4, n_people: int = 11) -> dict:
    rng = random.Random(seed)
    people = make_people(rng, n_people)
    (out / "events").mkdir(parents=True, exist_ok=True)

    # ---- events and invoices
    start = date(2027, 1, 10)
    events = []
    for e in range(n_events):
        attendees = rng.sample(people, rng.randint(5, min(8, len(people))))
        ev_start = start + timedelta(days=45 * e + rng.randint(0, 12))
        eid = f"e{e + 1}"
        invoices = []
        for i, (vendor, desc, (lo, hi)) in enumerate(rng.sample(TEMPLATES, rng.randint(6, 9))):
            who = rng.sample(attendees, rng.randint(2, len(attendees)))
            total = Decimal(rng.randint(lo, hi)) + Decimal(rng.choice([0, 0, 0.5, 0.25, 0.75]))
            if i == 0:
                who = attendees
            if rng.random() < 0.5 and 100 % len(who) == 0:
                shares = [{"person": p["id"], "percent": str(100 // len(who))} for p in who]
            else:  # uneven integer percents that add to 100 (one invoice per event adds to 90)
                target = 90 if i == 3 and e % 2 == 0 else 100
                cuts = sorted(rng.sample(range(1, target), len(who) - 1))
                parts = [b - a for a, b in zip([0] + cuts, cuts + [target])]
                shares = [{"person": p["id"], "percent": str(x)} for p, x in zip(who, parts)]
            invoices.append({
                "id": f"{eid}_{i + 1}", "date": str(ev_start + timedelta(days=rng.randint(0, 6))),
                "vendor": vendor.format(rng.choice(PLACES)), "description": desc,
                "total": f"{total:.2f}", "shares": shares, "note": ""})
        ev = {"id": eid, "name": rng.choice([n for n in EVENT_NAMES if n not in [x["name"] for x in events]]),
              "start_date": str(ev_start), "end_date": str(ev_start + timedelta(days=6)),
              "people": attendees, "invoices": invoices}
        events.append(ev)
        (out / "events" / f"{eid}.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
    (out / "payments.json").write_text("[]")

    from run import load_scenarios  # noqa: E402
    _, invoices, _ = load_scenarios(out)
    owed = {i.id: compute_obligations(i).owed for i in invoices}
    inv_by = {i.id: i for i in invoices}
    name_of = {p["id"]: p["name"] for p in people}
    twins = {p["id"] for p in people}

    # ---- payments
    payments, truth = [], []

    def emit(sender, cents, when, memo, payer, links, noise, pattern, how="synthetic", true_payer=None):
        pid = f"s{len(payments) + 1:03d}"
        payments.append({"id": pid, "sender_name": sender, "amount": f"{cents / 100:.2f}",
                         "timestamp": when.isoformat(timespec="minutes"), **({"memo": memo} if memo else {})})
        t = {"payment_id": pid, "payer": payer, "how": how, "noise": noise, "pattern": pattern,
             "links": [{"invoice": i, "owed_by": p, "amount": f"{c / 100:.2f}"} for i, p, c in links],
             "unmatched": "0.00"}
        if true_payer:
            t["true_payer"] = true_payer
        truth.append(t)
        return pid

    def word(inv_id):  # a keyword a person might type for this invoice
        d = inv_by[inv_id].description.lower().split()
        return rng.choice([w for w in d if len(w) > 3] or d)

    for ev in events:
        left = {}  # (person, invoice) -> cents still owed
        for inv in ev["invoices"]:
            for p, c in owed[inv["id"]].items():
                left[(p, inv["id"])] = c
        for person in ev["people"]:
            pid = person["id"]
            mine = sorted(((inv_by[i].date, i) for (p, i) in left if p == pid and left[(p, i)] > 0))
            while mine:
                mine = [(d, i) for d, i in mine if left[(pid, i)] > 0]
                if not mine:
                    break
                r = rng.random()
                when = datetime.combine(max(d for d, _ in mine) + timedelta(days=rng.randint(0, 9)),
                                        datetime.min.time()) + timedelta(hours=rng.randint(8, 21), minutes=rng.randint(0, 59))
                sender, noise = variant(person["name"], rng, twins)

                def pay(links, memo, pattern, payer=pid, snd=None, nz=None):
                    cents = sum(c for _, _, c in links)
                    for i, p, c in links:
                        left[(p, i)] -= c
                    emit(snd or sender, cents, when, memo, payer, links, nz or noise, pattern)
                    if rng.random() < 0.06:  # the bank also sends a notification of the same payment
                        emit(snd or sender, cents, when + timedelta(minutes=rng.randint(1, 4)),
                             "Chase alert: deposit received", None, [], "bank alert", "duplicate", "held", payer)

                if r < 0.30:  # one share, often with a keyword
                    d, i = rng.choice(mine)
                    pay([(i, pid, left[(pid, i)])], word(i) if rng.random() < .6 else "", "single share")
                elif r < 0.50 and len(mine) >= 2:  # several shares at once, no memo
                    k = rng.randint(2, min(4, len(mine)))
                    chosen = mine[:k] if rng.random() < 0.6 else rng.sample(mine, k)
                    pay([(i, pid, left[(pid, i)]) for _, i in chosen], "", "lump oldest" if chosen == mine[:k] else "lump random")
                elif r < 0.62 and len(mine) >= 2:  # itemised memo
                    chosen = rng.sample(mine, 2)
                    memo = " + ".join(f"{word(i)} {round(left[(pid, i)] / 100)}" for _, i in chosen)
                    pay([(i, pid, left[(pid, i)]) for _, i in chosen], memo, "itemised")
                elif r < 0.74:  # covering someone else on a shared invoice
                    cand = [(i, q["id"]) for _, i in mine for q in ev["people"]
                            if q["id"] != pid and left.get((q["id"], i), 0) > 0]
                    if not cand:
                        continue
                    i, other = rng.choice(cand)
                    both = [(i, other, left[(other, i)])]
                    if rng.random() < .6:
                        both.insert(0, (i, pid, left[(pid, i)]))
                    pay(both, f"covering {name_of[other].split()[0]}'s share of {word(i)}", "covering")
                elif r < 0.80:  # deposit before the invoice exists
                    d, i = rng.choice(mine)
                    part = max(left[(pid, i)] // 2, 100)
                    pay([(i, pid, part)], f"deposit for {ev['name'].split()[0].lower()}", "deposit")
                    when = when  # (timestamp stays after the invoice here; deposits are tested by the real data)
                elif r < 0.84:  # someone outside the group pays for this person
                    d, i = rng.choice(mine)
                    stranger = rng.choice(STRANGERS)
                    cents = left[(pid, i)]
                    left[(pid, i)] -= cents
                    emit(stranger, cents, when, f"for {person['name'].split()[0]}", None, [(i, pid, cents)],
                         "third party", "third party", "held", pid)
                elif r < 0.88:  # unreadable bank text
                    d, i = rng.choice(mine)
                    cents = left[(pid, i)]
                    left[(pid, i)] -= cents
                    emit(rng.choice(UNREADABLE), cents, when, "", None, [(i, pid, cents)], "unreadable", "unreadable",
                         "held", pid)
                elif r < 0.91:  # an outgoing payment appears in the feed
                    emit(sender, rng.randint(1500, 8000), when, f"You paid {name_of[rng.choice(ev['people'])['id']]}",
                         None, [], noise, "outgoing", "held")
                else:  # oldest-first partial lump
                    take = max(sum(left[(pid, i)] for _, i in mine) // 3, 500)
                    links, rest = [], take
                    for d, i in mine:
                        c = min(left[(pid, i)], rest)
                        if c:
                            links.append((i, pid, c))
                            rest -= c
                    pay(links, "", "partial lump")
    (out / "payments.json").write_text(json.dumps(payments, indent=1, ensure_ascii=False))
    (out / "expected.json").write_text(json.dumps({"payments": truth}, indent=1, ensure_ascii=False))
    return {"events": len(events), "people": len(people), "invoices": len(invoices), "payments": len(payments)}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(generate(seed, Path(__file__).resolve().parent.parent / "scenarios_synth" / f"seed{seed}"))
