# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compares harnesses from their eval results: pass rate and turns, paired on the same tasks."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

BOOTSTRAP_SAMPLES = 2000


def load_results(eval_dir: Path) -> Dict[str, List[dict]]:
    results = defaultdict(list)
    for file in sorted(Path(eval_dir).glob("*/*/result.json")):
        result = json.loads(file.read_text())
        results[result["harness"]].append(result)
    return dict(results)


def paired_ci(deltas: List[float], seed: int = 0) -> tuple:
    """Mean of `deltas` and its 95% bootstrap interval."""
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(BOOTSTRAP_SAMPLES))
    return statistics.fmean(deltas), means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]


def render_report(results: Dict[str, List[dict]], reference: str = None) -> str:
    names = list(results)
    reference = reference or ("base" if "base" in results else names[0])
    # Tasks every harness finished, so the rows compare like with like
    common = set.intersection(*({(r["task"], r["repeat"]) for r in rs} for rs in results.values()))
    by_key = {name: {(r["task"], r["repeat"]): r for r in rs if (r["task"], r["repeat"]) in common}
              for name, rs in results.items()}
    lines = [f"# Harness comparison ({len(common)} rollouts per harness on the same tasks)", "",
             "| harness | pass rate | turns (mean / median) | turns when solved | answered | "
             "timed out | tool errors / run | tokens / run |",
             "|---|---|---|---|---|---|---|---|"]
    for name in names:
        rows = list(by_key[name].values())
        if not rows:
            continue
        solved = [r["turns"] for r in rows if r["reward"] >= 1]
        tokens = statistics.fmean(r["prompt_tokens"] + r["completion_tokens"] for r in rows)
        lines.append(
            f"| {name} | {statistics.fmean(r['reward'] for r in rows):.1%} | "
            f"{statistics.fmean(r['turns'] for r in rows):.1f} / {statistics.median(r['turns'] for r in rows):.0f} | "
            f"{statistics.fmean(solved) if solved else float('nan'):.1f} | "
            f"{statistics.fmean(r['answered'] for r in rows):.0%} | {sum(r['timed_out'] for r in rows)} | "
            f"{statistics.fmean(r['tool_errors'] for r in rows):.2f} | {tokens:,.0f} |")
    lines.append("")

    for name in names:
        if name == reference or not common:
            continue
        keys = sorted(common)
        reward = paired_ci([by_key[name][k]["reward"] - by_key[reference][k]["reward"] for k in keys])
        turns = paired_ci([by_key[name][k]["turns"] - by_key[reference][k]["turns"] for k in keys])
        wins = sum(by_key[name][k]["reward"] > by_key[reference][k]["reward"] for k in keys)
        losses = sum(by_key[name][k]["reward"] < by_key[reference][k]["reward"] for k in keys)
        lines += [f"## {name} vs {reference}", "",
                  f"- pass rate: {reward[0]:+.1%} (95% CI {reward[1]:+.1%} to {reward[2]:+.1%}); "
                  f"solved only by {name}: {wins}, only by {reference}: {losses}",
                  f"- turns per task: {turns[0]:+.2f} (95% CI {turns[1]:+.2f} to {turns[2]:+.2f})", ""]

    for field in ("difficulty", "answer_kind"):
        groups = sorted({r[field] or "?" for rows in by_key.values() for r in rows.values()})
        lines += [f"## Pass rate and mean turns by {field}", "",
                  f"| {field} | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for group in groups:
            cells = []
            for name in names:
                rows = [r for r in by_key[name].values() if (r[field] or "?") == group]
                cells.append(f"{statistics.fmean(r['reward'] for r in rows):.0%} / "
                             f"{statistics.fmean(r['turns'] for r in rows):.1f}" if rows else "-")
            n = sum((r[field] or "?") == group for r in by_key[reference].values())
            lines.append(f"| {group} | {n} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
