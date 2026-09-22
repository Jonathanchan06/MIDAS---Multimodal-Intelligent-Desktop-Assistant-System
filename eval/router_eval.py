"""Router evaluation harness.

Runs core/orchestrator.py's Orchestrator.route() against a labeled test
set (eval/test_cases.json) and reports accuracy, per-route precision/
recall, and a confusion matrix -- so a change to a route description,
ROUTER_MODEL, or the router prompt has a number attached to it instead
of a vibe.

Usage:
    python eval/router_eval.py

Requires Ollama running with ROUTER_MODEL (see config.py) already pulled.

Work in progress still, can only do basic evals because json file is so small
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Running this file directly (not as `python -m eval.router_eval`) puts
# only eval/ on sys.path, not the project root -- add it so `core` and
# `main` are importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.orchestrator import Orchestrator 
from main import TOOL_REGISTRY  

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_eval() -> None:
    cases = load_test_cases()
    orchestrator = Orchestrator(tool_registry=TOOL_REGISTRY)
    valid_routes = {r["name"] for r in orchestrator._routes}

    # confusion[expected][predicted] += 1
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    misclassified = []

    print(f"Running {len(cases)} test cases against the router...\n")
    for i, case in enumerate(cases, start=1):
        text = case["text"]
        expected = case["expected_route"]
        if expected not in valid_routes:
            print(f"  WARNING: {expected!r} is not a real route (typo?) -- skipping {text!r}")
            continue

        predicted = orchestrator.route(text, [])
        confusion[expected][predicted] += 1
        mark = "OK" if predicted == expected else "MISS"
        print(f"[{i}/{len(cases)}] {mark:<4} expected={expected:<16} got={predicted:<16} {text!r}")
        if predicted != expected:
            misclassified.append((text, expected, predicted))

    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[r][r] for r in confusion)
    accuracy = correct / total if total else 0.0

    print(f"\n=== Overall: {correct}/{total} correct ({accuracy:.1%}) ===\n")

    print(f"{'route':<20}{'precision':<12}{'recall':<12}{'support':<10}")
    for route in sorted(valid_routes):
        support = sum(confusion[route].values())  # how many test cases expected this route
        predicted_count = sum(confusion[r][route] for r in confusion)  # how often it was predicted
        if support == 0 and predicted_count == 0:
            continue
        tp = confusion[route][route]
        precision = tp / predicted_count if predicted_count else float("nan")
        recall = tp / support if support else float("nan")
        print(f"{route:<20}{precision:<12.2f}{recall:<12.2f}{support:<10}")

    if misclassified:
        print("\n=== Misclassified ===")
        for text, expected, predicted in misclassified:
            print(f"  expected={expected!r:<18} got={predicted!r:<18} text={text!r}")
    else:
        print("\nAll test cases passed.")


if __name__ == "__main__":
    run_eval()
