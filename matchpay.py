#!/usr/bin/env python3
"""matchpay: one command for the whole system.

  python3 matchpay.py demo              fresh start on the sample data, writes a report (add --open)
  python3 matchpay.py run               match payments to invoices, write out/report.html
  python3 matchpay.py review            work through what the system could not settle
  python3 matchpay.py review --aliases  approve or reject proposed name aliases
  python3 matchpay.py doctor            check this machine is ready
  python3 matchpay.py pull-model        download the recommended local model (about 5 GB)
  python3 matchpay.py generate          make a new messy test scenario with known answers
  python3 matchpay.py score | stress | learn   evaluation tools (pass their own options after the name)

Add --llm to use the local model (Ollama); without it nothing needs any model. Everything
people decide and everything learned is kept in state/ (override with --state DIR).
"""

import argparse
import os
import runpy
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

RECOMMENDED_MODEL = "llama3.1:8b"
SMALL_MODEL = "qwen2.5:3b"


def _setup(args):
    if getattr(args, "state", None):
        os.environ["MATCH_STATE"] = str(Path(args.state).resolve())
    from algorithms import state
    from algorithms.dictionary import Dictionary
    from algorithms.identity import Directory
    from algorithms.llm_ollama import OllamaVoter
    from algorithms.reviews import ReviewStore
    from run import load_scenarios

    data = Path(args.data)
    if not (data / "events").is_dir():
        sys.exit(f"No scenario data in {data} (expected {data}/events/*.json and {data}/payments.json).")
    from algorithms.validate import validate
    errors, _ = validate(data)
    if errors:
        print(f"There are problems with the data in {data}:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print("The expected format is described in data-format.html and the README.", file=sys.stderr)
        sys.exit(1)
    events, invoices, payments = load_scenarios(data)
    return dict(
        events=events, invoices=invoices, payments=payments, data=data,
        dictionary=Dictionary(state.dictionary_path()), store=ReviewStore(),
        directory=Directory.build(events),
        llm=OllamaVoter() if getattr(args, "llm", False) else None,
    )


def _match(ctx):
    from matcher import match_detailed
    return match_detailed(ctx["events"], ctx["invoices"], ctx["payments"], ctx["dictionary"], ctx["llm"], ctx["store"].load())


def cmd_run(args) -> int:
    from algorithms.report import write_outputs
    from algorithms.review import ROUNDING_CENTS
    ctx = _setup(args)
    result, detail = _match(ctx)
    out = Path(args.out)
    report = write_outputs(out, f"Payment matching: {ctx['data'].name}", result, detail, ctx["events"], ctx["invoices"],
                           ctx["payments"], ctx["directory"], ctx["llm"].model if ctx["llm"] else None)
    placed = sum(l.amount_cents for l in result.links)
    needs = [u for u in result.unmatched if u.amount_cents >= ROUNDING_CENTS and not u.reason.startswith("reviewed:")]
    paid = sum(1 for i in result.invoices if str(i.status) == "paid")
    print(f"{len(ctx['payments'])} payments, {len(ctx['invoices'])} invoices")
    print(f"  matched   ${placed / 100:>10,.2f}  in {len(result.links)} links; {paid}/{len(result.invoices)} invoices fully paid")
    print(f"  waiting   ${sum(u.amount_cents for u in needs) / 100:>10,.2f}  in {len(needs)} item(s) for a person"
          + ("   ->  python3 matchpay.py review" if needs else ""))
    if detail.proposals:
        print(f"  {detail.proposals} new name alias(es) proposed   ->  python3 matchpay.py review --aliases")
    if ctx["llm"]:
        print(f"  local model {ctx['llm'].model}: {ctx['llm'].calls} new question(s), {ctx['llm'].hits} answered from cache")
    print(f"report: {report}")
    if args.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


def cmd_review(args) -> int:
    from algorithms import review as R
    ctx = _setup(args)
    if args.apply:
        n = R.apply_file(args.apply, ctx["store"], ctx["dictionary"])
        print(f"recorded {n} decision(s) from {args.apply}")
        return 0
    if args.aliases:
        n = R.review_aliases(ctx["dictionary"], ctx["directory"])
        print(f"{n} alias decision(s) recorded. They take effect on the next run.")
        return 0
    result, detail = _match(ctx)
    queue = R.build_queue(result, detail, ctx["payments"], ctx["directory"], ctx["store"].load())
    if not queue:
        print("Nothing is waiting for a person.")
        return 0
    if args.list:
        for n, item in enumerate(queue, 1):
            R.show(item, n, len(queue))
        return 0
    print(f"{len(queue)} item(s) to look at. Your decisions are saved as you go and used on the next run.")
    n = R.review_payments(queue, ctx["store"], ctx["dictionary"], ctx["directory"], {i.id for i in ctx["invoices"]})
    print(f"\n{n} decision(s) recorded. Run  python3 matchpay.py run  to apply them.")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    fresh = out / "state"
    shutil.rmtree(fresh, ignore_errors=True)  # nothing carries over from earlier runs
    args.state = str(fresh)
    print("Demo: starting fresh (no earlier decisions or learned aliases).\n")
    rc = cmd_run(args)
    print(f"\nTry it:   python3 matchpay.py review --state {fresh}\n          python3 matchpay.py run --state {fresh} --out {out}")
    return rc


def cmd_generate(args) -> int:
    from algorithms.synth import generate
    out = Path(args.out or ROOT / "scenarios_synth" / f"seed{args.seed}")
    info = generate(args.seed, out, args.events, args.people)
    print(f"wrote {out}: {info}\nscore it:  python3 matchpay.py score --data {out}")
    return 0


def cmd_doctor(args) -> int:
    ok = True

    def line(good, text, fix=""):
        nonlocal ok
        ok = ok and good
        print(f"  {'ok ' if good else 'FIX'}  {text}" + (f"\n         {fix}" if fix and not good else ""))

    print("Checking this machine:")
    line(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]} (needs 3.11 or newer)", "install a newer Python")
    data = Path(args.data)
    from algorithms.validate import validate
    errors, warnings = validate(data)
    line(not errors, f"scenario data in {data}/ is valid" + (f" ({len(warnings)} note(s))" if warnings else ""),
         "; ".join(errors[:3]))
    try:
        probe = ROOT / "state"
        probe.mkdir(exist_ok=True)
        (probe / ".w").write_text("x")
        (probe / ".w").unlink()
        line(True, "state/ folder is writable")
    except OSError:
        line(False, "state/ folder is writable", "check folder permissions")
    exe = shutil.which("ollama")
    print("\nLocal model (optional; everything works without it, with a few more senders left for review):")
    if not exe:
        print("  --   Ollama is not installed. Get it from https://ollama.com, then:  python3 matchpay.py pull-model")
        return 0 if ok else 1
    print(f"  ok   Ollama found at {exe}")
    try:
        tags = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).read().decode()
        up = True
    except (urllib.error.URLError, OSError):
        tags, up = "", False
    print(f"  {'ok ' if up else '--  '} Ollama server is {'running' if up else 'not running (start the Ollama app; it also starts on first use)'}")
    if up:
        for m, why in ((RECOMMENDED_MODEL, "recommended, about 5 GB"), (SMALL_MODEL, "lighter alternative, about 2 GB")):
            have = m in tags
            print(f"  {'ok ' if have else '--  '} {m} ({why}): {'installed' if have else 'not installed  ->  python3 matchpay.py pull-model' + (' --small' if m == SMALL_MODEL else '')}")
    return 0 if ok else 1


def cmd_pull(args) -> int:
    model = SMALL_MODEL if args.small else RECOMMENDED_MODEL
    if not shutil.which("ollama"):
        sys.exit("Ollama is not installed. Get it from https://ollama.com first.")
    print(f"Downloading {model} ({'about 2 GB' if args.small else 'about 5 GB'})...")
    return subprocess.call(["ollama", "pull", model])


PASSTHROUGH = {"score": "algorithms.score", "stress": "algorithms.robustness", "learn": "algorithms.learn"}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in PASSTHROUGH:  # these tools have their own options
        module = PASSTHROUGH[sys.argv[1]]
        sys.argv = [module] + sys.argv[2:]
        runpy.run_module(module, run_name="__main__")
        return 0
    ap = argparse.ArgumentParser(prog="matchpay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, llm=True):
        p.add_argument("--data", default=str(ROOT / "scenarios"), help="scenario folder (default: scenarios/)")
        p.add_argument("--state", help="folder for decisions and learned data (default: state/)")
        if llm:
            p.add_argument("--llm", action="store_true", help="use the local model (Ollama)")

    for name, fn, helptext in (("run", cmd_run, "match and write the report"), ("demo", cmd_demo, "fresh run for showing it off")):
        p = sub.add_parser(name, help=helptext)
        common(p)
        p.add_argument("--out", default=str(ROOT / "out"), help="where to write report.html and friends")
        p.add_argument("--open", action="store_true", help="open the report in a browser")
        p.set_defaults(fn=fn)
    p = sub.add_parser("review", help="decide what the system could not")
    common(p)
    p.add_argument("--aliases", action="store_true", help="approve or reject proposed name aliases")
    p.add_argument("--list", action="store_true", help="just show the queue")
    p.add_argument("--apply", metavar="FILE", help="apply decisions from a JSON file")
    p.set_defaults(fn=cmd_review)
    p = sub.add_parser("generate", help="make a messy test scenario with known answers")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--events", type=int, default=4)
    p.add_argument("--people", type=int, default=11)
    p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("doctor", help="check this machine")
    common(p, llm=False)
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("pull-model", help="download the recommended local model")
    p.add_argument("--small", action="store_true", help=f"get {SMALL_MODEL} (about 2 GB) instead")
    p.set_defaults(fn=cmd_pull)
    for name, module in PASSTHROUGH.items():  # listed for --help; handled before argparse in main()
        sub.add_parser(name, help=f"python3 -m {module}", add_help=False)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
