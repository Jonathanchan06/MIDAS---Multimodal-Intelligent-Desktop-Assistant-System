"""Argument-extraction evaluation harness.

The router eval (eval/router_eval.py) checks *which* tool a message goes
to. This checks the next step: given the right tool, does the extraction
call (Orchestrator.extract_raw -- same prompt, schema bounding, and
sampling as a live request) pull out the right arguments?

It scores the model's RAW output, before any post_extract guard runs.
The guards in tools/wardrobe.py exist because the model gets these
wrong; this measures how often, so a model swap (qwen3:8b vs qwen2.5:7b)
or a purpose-trained extractor (the LoRA) has a number attached instead
of a vibe. A good extractor should pass without needing the guards.

Every failure is tagged with one category, matching the failure modes
seen in practice (see CLAUDE.md "Extraction design rule"):

    invalid_output    not JSON (e.g. a generation truncated by its cap)
    invented_id       a number the user never said (item_id 12345)
    extra_field       a field nobody asked to change (category on a
                      layer-only update) -- any present key is a change
    padded            free text not grounded in the user's words
                      ("daily wear/office work" for "what should I wear")
    wrong_value       wrong enum/number/words
    wrong_time        add_reminder's slots resolve to the wrong moment
    unparseable_time  slots the reminders tool can't resolve at all
    missing           a required value or list entry is absent
    extra_entry       a list entry nobody asked for

Usage:
    python eval/extraction_eval.py                      # config's model
    python eval/extraction_eval.py --model qwen2.5:7b --temperature 0.3
    python eval/extraction_eval.py --tool suggest_fit --repeats 3
    python eval/extraction_eval.py --out eval/results/qwen3.json
    python eval/extraction_eval.py --compare a.json b.json

Requires Ollama running with the model pulled. Cases live in
eval/extraction_cases.json; the matcher spec is documented in check().
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Running this file directly (not as `python -m eval.extraction_eval`)
# puts only eval/ on sys.path, not the project root -- add it so `core`
# and `main` are importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.orchestrator import Orchestrator, build_extraction_messages  # noqa: E402
from main import TOOL_REGISTRY  # noqa: E402
from tools.reminders import _parse_datetime, resolve_when  # noqa: E402

CASES_PATH = Path(__file__).parent / "extraction_cases.json"
# "core": the original cases, one clear phrasing each. "hard": realistic
# messages templates don't produce — spoken/typo'd text, spelled-out
# times, self-corrections, distractors, unfamiliar garment words. Written
# before the LoRA was trained and kept frozen: training data must never be
# written against it, or it stops measuring generalization.
CASE_SETS = {
    "core": CASES_PATH,
    "hard": Path(__file__).parent / "extraction_cases_hard.json",
}
# Pinned clock for every case: a Wednesday 10:00, so "tonight at 9pm",
# "at noon", and weekday cases have one right answer whenever this runs.
EVAL_NOW = "2026-11-25T10:00:00"

CATEGORIES = [
    "invalid_output", "invented_id", "extra_field", "padded", "wrong_value",
    "wrong_time", "unparseable_time", "missing", "extra_entry",
]

# Words that never make free text "ungrounded" on their own.
_FILLER = {
    "a", "an", "the", "my", "me", "i", "to", "for", "of", "and", "on", "in",
    "at", "with", "some", "this", "that", "it", "is", "be", "or", "your",
}
# Words that only point at "the outfit you just suggested" (mark_worn).
_DEICTIC = _FILLER | {
    "those", "these", "them", "outfit", "fit", "everything", "all", "clothe",
    "suggestion", "look", "one", "same", "suggested", "picked", "you", "what",
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TIME_TOLERANCE = timedelta(minutes=3)


# --- text helpers -----------------------------------------------------------

def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _tokens(text) -> list[str]:
    text = str(text or "").lower().replace("'s", "").replace("’s", "")
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text)]


def _has_word(value: str, word: str) -> bool:
    """word starts a word in value (prefixes allowed: 'sneaker' hits 'sneakers')."""
    return re.search(r"\b" + re.escape(word.lower()), str(value).lower()) is not None


def _grounded(tok: str, said: set[str]) -> bool:
    if tok in said:
        return True
    # Allow inflection either way ('interviewing' ~ 'interview').
    return any(len(min(tok, s, key=len)) >= 4 and (s.startswith(tok) or tok.startswith(s)) for s in said)


def _is_empty(value) -> bool:
    return value is None or value == "" or value == [] or value == 0


# --- matching ---------------------------------------------------------------

class Ctx:
    def __init__(self, text: str, history: list[dict], now: datetime):
        self.text = text
        # The model may legitimately copy from earlier turns (e.g. an id
        # shown in a reminders list), so history counts as "said".
        self.said_text = " ".join([text, *(m.get("content", "") for m in history)])
        self.said = set(_tokens(self.said_text))
        self.now = now


def _fail(out: list, category: str, path: str, message: str) -> None:
    out.append({"category": category, "path": path, "message": message})


def check(spec, value, ctx: Ctx, path: str, present: bool = True) -> list[dict]:
    """Returns a list of failures (empty = pass). Spec forms:

    scalar                     equality (strings case-insensitive)
    ["w1", "w2"]               shorthand for {"words": [...], "grounded": true}
    {"eq": v} / {"one_of": [...]}
    {"words": [...]}           each must start a word in the value
    {"no_words": [...]}        none may appear (fails as padded)
    {"grounded": true}         every non-filler word was said by the user
                               (an unsaid number fails as invented_id)
    {"empty": true}            None / "" / [] / 0 / absent
    {"optional": true}         absent is fine; other keys apply if present
    {"id_from_text": true}     an integer the user actually said
    {"deictic": true}          a list that's empty or only says "that/it/the outfit"
    {"time": {...}}            remind_at; see _check_time
    {"entries": [...], "only": bool}  list matched order-free, no extras
    {"fields": {...}, "only": bool}   object; "only" makes other keys extra_field
    {"any_of": [spec, ...]}    passes if any sub-spec passes (for answers
                               that are equally correct at runtime)
    """
    out: list[dict] = []
    if isinstance(spec, list):
        spec = {"words": spec, "grounded": True}
    if not isinstance(spec, dict):
        spec = {"eq": spec}
    if "any_of" in spec:
        tries = [check(s, value, ctx, path, present) for s in spec["any_of"]]
        return [] if any(not t for t in tries) else tries[0]

    if not present:
        if spec.get("optional") or spec.get("empty") or spec.get("deictic"):
            return out
        _fail(out, "missing", path, "absent")
        return out

    if "eq" in spec and not _equal(value, spec["eq"]):
        # A wrong integer the user never said (reminder_id 2126112518014500
        # for "reminder 2") is an invented id, not a misread one.
        unsaid_int = (
            isinstance(value, int) and not isinstance(value, bool)
            and not re.search(rf"(?<!\d){value}(?!\d)", ctx.said_text)
        )
        _fail(out, "invented_id" if unsaid_int else "wrong_value", path, f"{value!r} != {spec['eq']!r}")
    if "one_of" in spec and not any(_equal(value, v) for v in spec["one_of"]):
        _fail(out, "wrong_value", path, f"{value!r} not in {spec['one_of']!r}")
    if spec.get("empty") and not _is_empty(value):
        _fail(out, "padded", path, f"expected empty, got {value!r}")

    if any(k in spec for k in ("words", "no_words", "grounded")):
        if not isinstance(value, str):
            _fail(out, "wrong_value", path, f"expected text, got {value!r}")
            return out
        missing = [w for w in spec.get("words", []) if not _has_word(value, w)]
        if missing:
            category = "missing" if not value.strip() else "wrong_value"
            _fail(out, category, path, f"{value!r} lacks {missing}")
        extra = [w for w in spec.get("no_words", []) if _has_word(value, w)]
        if extra:
            _fail(out, "padded", path, f"{value!r} restates {extra}")
        if spec.get("grounded"):
            unsaid = [t for t in _tokens(value) if t not in _FILLER and not _grounded(t, ctx.said)]
            if any(t.isdigit() for t in unsaid):
                _fail(out, "invented_id", path, f"{value!r}: number {unsaid} never said")
            elif unsaid:
                _fail(out, "padded", path, f"{value!r}: {unsaid} never said")

    if spec.get("id_from_text"):
        if not isinstance(value, int) or isinstance(value, bool) or not re.search(
            rf"(?<!\d){value}(?!\d)", ctx.said_text
        ):
            _fail(out, "invented_id", path, f"{value!r} was never said")

    if spec.get("deictic"):
        refs = value if isinstance(value, list) else [value]
        named = [r for r in refs if set(_tokens(r)) - _DEICTIC]
        if named:
            _fail(out, "wrong_value", path, f"{named!r} for 'the suggested outfit'")

    if "time" in spec:
        out += _check_time(spec["time"], value, ctx, path)
    if "entries" in spec:
        out += _check_entries(spec, value, ctx, path)
    if "fields" in spec:
        out += _check_fields(spec, value, ctx, path)
    return out


def _equal(value, expected) -> bool:
    if isinstance(expected, str):
        return isinstance(value, str) and value.strip().lower() == expected.lower()
    if isinstance(expected, bool) or isinstance(value, bool):
        return value is expected
    return value == expected


def _check_fields(spec: dict, value, ctx: Ctx, path: str) -> list[dict]:
    out: list[dict] = []
    if not isinstance(value, dict):
        _fail(out, "wrong_value", path, f"expected object, got {value!r}")
        return out
    fields = spec["fields"]
    for key, sub in fields.items():
        out += check(sub, value.get(key), ctx, f"{path}.{key}", present=key in value)
    if spec.get("only"):
        for key in value.keys() - fields.keys():
            _fail(out, "extra_field", f"{path}.{key}", f"unrequested {key}={value[key]!r}")
    return out


def _check_entries(spec: dict, value, ctx: Ctx, path: str) -> list[dict]:
    """Order-free list match: each expected entry claims the unused actual
    entry it fits best; anything left over is an extra_entry."""
    out: list[dict] = []
    if not isinstance(value, list):
        _fail(out, "wrong_value", path, f"expected list, got {value!r}")
        return out
    unused = list(range(len(value)))
    for i, sub in enumerate(spec["entries"]):
        if not unused:
            _fail(out, "missing", f"{path}[{i}]", f"no entry for {sub!r}")
            continue
        scored = [(check(sub, value[j], ctx, f"{path}[{j}]"), j) for j in unused]
        failures, best = min(scored, key=lambda pair: len(pair[0]))
        unused.remove(best)
        out += failures
    for j in unused:
        _fail(out, "extra_entry", f"{path}[{j}]", f"unrequested entry {value[j]!r}")
    return out


def _check_time(spec: dict, value, ctx: Ctx, path: str) -> list[dict]:
    """Time spec, all relative to the moment of the extraction call:
    {"in_minutes": n}             now + n (± 3 min)
    {"at": "HH:MM"}               next occurrence of that clock time
    {"at", "day_offset": n}       that time n days from today
    {"day_offset": n}             any time on that day
    {"weekday": "friday"}         the coming one, or the week after
                                  ("next Friday" is genuinely ambiguous)
    {"date": "MM-DD"}             next occurrence of that date
    {"between": ["HH:MM","HH:MM"]}  time-of-day range ("morning")
    """
    out: list[dict] = []
    try:
        got = _parse_datetime(str(value))
    except Exception:
        _fail(out, "unparseable_time", path, f"reminders tool can't resolve {value!r}")
        return out
    if got.tzinfo is not None:
        got = got.astimezone().replace(tzinfo=None)

    now = ctx.now
    if "in_minutes" in spec:
        target = now + timedelta(minutes=spec["in_minutes"])
        if abs(got - target) > _TIME_TOLERANCE:
            _fail(out, "wrong_time", path, f"{got:%a %b %d %H:%M}, wanted ~{target:%a %b %d %H:%M}")
        return out

    days: list = []
    if "day_offset" in spec:
        days = [(now + timedelta(days=spec["day_offset"])).date()]
    elif "weekday" in spec:
        ahead = (_WEEKDAYS.index(spec["weekday"]) - now.weekday()) % 7 or 7
        days = [(now + timedelta(days=ahead + extra)).date() for extra in (0, 7)]
    elif "date" in spec:
        month, day = map(int, spec["date"].split("-"))
        this_year = now.replace(month=month, day=day).date()
        days = [this_year if this_year >= now.date() else this_year.replace(year=now.year + 1)]
    elif "at" in spec:
        hour, minute = map(int, spec["at"].split(":"))
        today_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        days = [(today_at if today_at > now else today_at + timedelta(days=1)).date()]

    if days and got.date() not in days:
        wanted = " or ".join(f"{d:%a %b %d}" for d in days)
        _fail(out, "wrong_time", path, f"{got:%a %b %d %H:%M}, wanted {wanted}")
    if "at" in spec and got.strftime("%H:%M") != spec["at"]:
        _fail(out, "wrong_time", path, f"{got:%H:%M}, wanted {spec['at']}")
    if "between" in spec:
        low, high = spec["between"]
        if not low <= got.strftime("%H:%M") <= high:
            _fail(out, "wrong_time", path, f"{got:%H:%M}, wanted {low}-{high}")
    return out


# --- running ----------------------------------------------------------------

def load_cases(tool_filter: str | None, which: str = "core") -> list[dict]:
    names = list(CASE_SETS) if which == "all" else [which]
    cases = []
    for name in names:
        with open(CASE_SETS[name], encoding="utf-8") as f:
            cases += [{**case, "set": name} for case in json.load(f)]
    unknown = {c["tool"] for c in cases} - TOOL_REGISTRY.keys()
    if unknown:
        sys.exit(f"unknown tool(s) in case files: {sorted(unknown)}")
    if tool_filter:
        cases = [c for c in cases if c["tool"] == tool_filter]
        if not cases:
            sys.exit(f"no cases for tool {tool_filter!r}")
    return cases


def run_case(orchestrator: Orchestrator, case: dict, now: datetime) -> dict:
    function_schema = TOOL_REGISTRY[case["tool"]]["schema"]["function"]
    history = case.get("history", [])
    record = {"tool": case["tool"], "set": case.get("set", "core"), "text": case["text"],
              "raw": None, "args": None}
    start = time.perf_counter()
    try:
        response = orchestrator.extract_raw(function_schema, case["text"], history, now=now)
    except Exception as exc:
        record.update(seconds=time.perf_counter() - start, failures=[
            {"category": "invalid_output", "path": "", "message": f"call failed: {exc!r}"}
        ])
        return record
    record["seconds"] = time.perf_counter() - start
    record["raw"] = response.message.content
    record["output_tokens"] = response.eval_count
    record["gen_seconds"] = (response.eval_duration or 0) / 1e9
    record["load_seconds"] = (response.load_duration or 0) / 1e9

    try:
        args = json.loads(response.message.content)
    except json.JSONDecodeError:
        record["failures"] = [{
            "category": "invalid_output", "path": "",
            "message": f"not JSON ({response.eval_count} tokens, likely hit the cap)",
        }]
        return record
    record["args"] = args
    view = TOOL_VIEWS.get(case["tool"], lambda a, _now: a)(args, now)
    spec = {"fields": case["expect"], "only": True}
    record["failures"] = check(spec, view, Ctx(case["text"], history, now), "args")
    return record


def _reminder_view(args: dict, now: datetime) -> dict:
    """add_reminder extracts day/time/in_minutes slots; what matters is
    the moment they resolve to, so grade that (as remind_at) through the
    tool's own resolver rather than the exact slot values — 'tonight at
    9pm' is equally right as day='today' or day=''."""
    if not isinstance(args, dict):
        return args
    view = {k: v for k, v in args.items() if k not in ("day", "time", "in_minutes", "date")}
    try:
        view["remind_at"] = resolve_when(
            args.get("day", ""), args.get("time", ""), args.get("in_minutes", 0),
            args.get("date", ""), now=now,
        ).isoformat()
    except Exception as exc:
        view["remind_at"] = f"<unresolvable: {exc}>"
    return view


# Per-tool transform from raw extraction to what the cases grade.
TOOL_VIEWS = {"add_reminder": _reminder_view}


def summarize(records: list[dict], cases: list[dict]) -> dict:
    gaps = {c["text"] for c in cases if c.get("known_gap")}
    passed = [r for r in records if not r["failures"]]
    scored = [r for r in records if r["text"] not in gaps]

    per_tool: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0, "seconds": []})
    for r in records:
        t = per_tool[r["tool"]]
        t["total"] += 1
        t["passed"] += not r["failures"]
        t["seconds"].append(r["seconds"])

    per_set: dict[str, dict] = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in records:
        per_set[r.get("set", "core")]["total"] += 1
        per_set[r.get("set", "core")]["passed"] += not r["failures"]

    # Count each category once per run, so one bad list doesn't dominate.
    categories = Counter(c for r in records for c in {f["category"] for f in r["failures"]})
    timed = [r for r in records if "output_tokens" in r]
    seconds = sorted(r["seconds"] for r in timed)
    tokens = sum(r["output_tokens"] or 0 for r in timed)
    gen = sum(r["gen_seconds"] for r in timed)
    return {
        "passed": len(passed),
        "total": len(records),
        "passed_excluding_known_gaps": sum(not r["failures"] for r in scored),
        "total_excluding_known_gaps": len(scored),
        "per_set": dict(sorted(per_set.items())),
        "per_tool": {
            tool: {
                "passed": t["passed"], "total": t["total"],
                "mean_seconds": round(statistics.mean(t["seconds"]), 3),
            }
            for tool, t in sorted(per_tool.items())
        },
        "failure_categories": {c: categories.get(c, 0) for c in CATEGORIES},
        "latency": {
            "mean_seconds": round(statistics.mean(seconds), 3) if seconds else None,
            "p50_seconds": round(seconds[len(seconds) // 2], 3) if seconds else None,
            "p95_seconds": round(seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))], 3) if seconds else None,
            "mean_output_tokens": round(tokens / len(timed), 1) if timed else None,
            "tokens_per_second": round(tokens / gen, 1) if gen else None,
        },
    }


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "0/0"


def print_summary(summary: dict, records: list[dict], repeats: int) -> None:
    print("\n" + "=" * 72)
    print(f"Overall:                  {_pct(summary['passed'], summary['total'])}")
    print(f"Excluding known gaps:     {_pct(summary['passed_excluding_known_gaps'], summary['total_excluding_known_gaps'])}")
    if len(summary["per_set"]) > 1:
        for name, s in summary["per_set"].items():
            print(f"  {name + ' set:':<24}{_pct(s['passed'], s['total'])}")

    print(f"\n{'tool':<24} {'pass':>14} {'mean s':>8}")
    for tool, t in summary["per_tool"].items():
        print(f"{tool:<24} {_pct(t['passed'], t['total']):>14} {t['mean_seconds']:>8.2f}")

    print("\nFailure categories (runs with at least one):")
    for category, count in summary["failure_categories"].items():
        if count:
            print(f"  {category:<18} {count}")

    lat = summary["latency"]
    print(
        f"\nLatency: mean {lat['mean_seconds']}s, p50 {lat['p50_seconds']}s, "
        f"p95 {lat['p95_seconds']}s; {lat['mean_output_tokens']} output tokens/call "
        f"at {lat['tokens_per_second']} tok/s"
    )

    if repeats > 1:
        by_case: dict[tuple, list[bool]] = defaultdict(list)
        for r in records:
            by_case[(r["tool"], r["text"])].append(not r["failures"])
        flaky = [(k, v) for k, v in by_case.items() if any(v) and not all(v)]
        if flaky:
            print(f"\nFlaky ({len(flaky)} cases passed some runs, not all):")
            for (tool, text), results in flaky:
                print(f"  {sum(results)}/{len(results)}  {tool:<22} {text!r}")


def compare(path_a: str, path_b: str) -> None:
    """Side-by-side of two --out files: per-tool pass rates, latency, and
    which cases flipped. For before/after (LoRA) and per-profile reports."""
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    name_a, name_b = a["model"], b["model"]
    if name_a == name_b:
        name_a, name_b = Path(path_a).stem, Path(path_b).stem
    sa, sb = a["summary"], b["summary"]
    width = max(16, len(name_a), len(name_b))

    print(f"{'':<24} {name_a:>{width}} {name_b:>{width}}")
    print(f"{'overall':<24} {_pct(sa['passed'], sa['total']):>{width}} {_pct(sb['passed'], sb['total']):>{width}}")
    sets = sorted(set(sa.get("per_set", {})) | set(sb.get("per_set", {})))
    if len(sets) > 1 or a.get("cases") != b.get("cases"):
        for name in sets:
            pa, pb = sa.get("per_set", {}).get(name), sb.get("per_set", {}).get(name)
            cell = lambda p: _pct(p["passed"], p["total"]) if p else "-"  # noqa: E731
            print(f"  {name + ' set':<22} {cell(pa):>{width}} {cell(pb):>{width}}")
    for tool in sorted(set(sa["per_tool"]) | set(sb["per_tool"])):
        ta, tb = sa["per_tool"].get(tool), sb["per_tool"].get(tool)
        cell = lambda t: _pct(t["passed"], t["total"]) if t else "-"  # noqa: E731
        print(f"  {tool:<22} {cell(ta):>{width}} {cell(tb):>{width}}")
    for category in CATEGORIES:
        ca, cb = sa["failure_categories"].get(category, 0), sb["failure_categories"].get(category, 0)
        if ca or cb:
            print(f"  fail: {category:<16} {ca:>{width}} {cb:>{width}}")
    for key in ("mean_seconds", "p95_seconds", "mean_output_tokens", "tokens_per_second"):
        print(f"{key:<24} {sa['latency'][key]!s:>{width}} {sb['latency'][key]!s:>{width}}")

    # Flips only make sense on single-repeat runs of the same cases.
    if a["repeats"] == 1 and b["repeats"] == 1:
        ra = {(r["tool"], r["text"]): not r["failures"] for r in a["records"]}
        rb = {(r["tool"], r["text"]): not r["failures"] for r in b["records"]}
        for label, want in (("Fixed", True), ("Broke", False)):
            flips = [k for k in ra.keys() & rb.keys() if ra[k] != want and rb[k] == want]
            if flips:
                print(f"\n{label} in {name_b} ({len(flips)}):")
                for tool, text in sorted(flips):
                    print(f"  {tool:<22} {text!r}")


def regrade(path: str) -> None:
    """Re-check a results file's raw outputs with the current matcher and
    cases -- no model calls. Records that failed before parsing (call
    errors, non-JSON) keep their failures."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = load_cases(None, data.get("cases", "core"))
    by_key = {(c["set"], c["tool"], c["text"]): c for c in cases}
    now = datetime.fromisoformat(data["eval_now"])
    for record in data["records"]:
        case = by_key.get((record.get("set", "core"), record["tool"], record["text"]))
        if case is None or record.get("args") is None:
            continue
        view = TOOL_VIEWS.get(case["tool"], lambda a, _now: a)(record["args"], now)
        record["failures"] = check({"fields": case["expect"], "only": True}, view,
                                   Ctx(case["text"], case.get("history", []), now), "args")
    data["summary"] = summarize(data["records"], cases)
    data["regraded_at"] = datetime.now().isoformat(timespec="seconds")
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"=== {path} ({data['model']})")
    print_summary(data["summary"], data["records"], data["repeats"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="override config.EXTRACTION_MODEL")
    parser.add_argument("--num-gpu", type=int, help="layers on GPU (0 = CPU only, like the router)")
    parser.add_argument("--temperature", type=float, help="override sampling temperature")
    parser.add_argument("--tool", help="only run cases for this tool")
    parser.add_argument("--cases", choices=[*CASE_SETS, "all"], default="core",
                        help="which case set to run (default: core)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per case (sampling isn't greedy)")
    parser.add_argument("--out", help="write full results JSON here (for --compare)")
    parser.add_argument(
        "--now", default=EVAL_NOW,
        help=f"the clock the model and grader see (default {EVAL_NOW}, a Wednesday "
             "morning) -- pinned so runs at different times of day compare",
    )
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    parser.add_argument("--regrade", nargs="+", metavar="RESULT.json",
                        help="re-check saved raw outputs against the current case files "
                             "(after fixing a gold answer) and rewrite the results in place")
    parser.add_argument("--dump-prompts", metavar="PROMPTS.json",
                        help="write the exact extraction messages for each case (pinned clock) "
                             "and exit -- input for training/hf_generate.py")
    args = parser.parse_args()
    # Windows consoles/redirects default to cp1252, which can't print some
    # case texts (emoji in the hard set) and would crash mid-run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.compare:
        compare(*args.compare)
        return
    if args.regrade:
        for path in args.regrade:
            regrade(path)
        return

    if args.model:
        config.EXTRACTION_MODEL = args.model
    if args.num_gpu is not None:
        config.EXTRACTION_NUM_GPU = args.num_gpu
    if args.temperature is not None:
        config.ORCHESTRATOR_SAMPLING = {**config.ORCHESTRATOR_SAMPLING, "temperature": args.temperature}

    cases = load_cases(args.tool, args.cases)
    if args.dump_prompts:
        now = datetime.fromisoformat(args.now)
        prompts = [{"tool": c["tool"], "set": c["set"], "text": c["text"],
                    "messages": build_extraction_messages(
                        TOOL_REGISTRY[c["tool"]]["schema"]["function"], c["text"],
                        c.get("history", []), now=now)} for c in cases]
        Path(args.dump_prompts).write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(prompts)} prompts to {args.dump_prompts}")
        return
    orchestrator = Orchestrator(tool_registry=TOOL_REGISTRY)
    now = datetime.fromisoformat(args.now)

    print(f"Model {config.EXTRACTION_MODEL}, sampling {config.ORCHESTRATOR_SAMPLING}, "
          f"think={config.ORCHESTRATOR_THINK}, num_gpu={config.EXTRACTION_NUM_GPU}, now={now:%a %Y-%m-%d %H:%M}")
    # Warm-up: the first call pays the model load (~8.5s); keep it out of
    # the latency numbers.
    warm = run_case(orchestrator, {"tool": "list_wardrobe_items", "text": "list my shoes", "expect": {}}, now)
    print(f"Warm-up call: {warm['seconds']:.1f}s (load {warm.get('load_seconds', 0):.1f}s)\n")

    records = []
    total = len(cases) * args.repeats
    for n, (case, _) in enumerate(((c, r) for r in range(args.repeats) for c in cases), start=1):
        record = run_case(orchestrator, case, now)
        records.append(record)
        mark = "OK" if not record["failures"] else "FAIL"
        print(f"[{n}/{total}] {mark:<4} {record['seconds']:5.2f}s {case['tool']:<22} {case['text']!r}")
        if record["failures"]:
            print(f"           -> {record['raw']}")
            for f in record["failures"]:
                print(f"              {f['category']:<16} {f['path']}: {f['message']}")

    summary = summarize(records, cases)
    print_summary(summary, records, args.repeats)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "model": config.EXTRACTION_MODEL,
            "sampling": config.ORCHESTRATOR_SAMPLING,
            "repeats": args.repeats,
            "cases": args.cases,
            "eval_now": now.isoformat(timespec="seconds"),
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "records": records,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
