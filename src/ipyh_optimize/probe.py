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
"""Probe rollouts, Harness²'s Probe step: run harnesses on a few train tasks and show the editors how
they behaved, never whether they were right (the rollouts are not graded).

    evidence/probe/<name>.md      digest of one harness's rollouts: per-task tool sequence, errors,
                                  turns after the answer was written, nudges, timeouts
    evidence/probe/CONTRAST.md    reference vs candidates, task by task (written for the composer)
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from ipyh.constants import NUDGE
from ipyh_optimize.eval import TOOL_ERROR, evaluate
from ipyh_optimize.tasks import Task

STEP_CHARS = 160  # code and output shown per tool call in a digest
ANSWER_WRITE = re.compile(r"answer[^\n]*(['\"]w['\"]|>|writefile|\.write\(|write_text)|(['\"]w['\"]|>|writefile)[^\n]*answer")


def run_probe(harnesses: Dict[str, Path], tasks: List[Task], out: Path, cache: Path, settings: dict,
              private: List[Path] = ()) -> List[dict]:
    """Ungraded rollouts of each harness on `tasks` (`settings`: base_url, model, api_key_env, concurrency)."""
    return asyncio.run(evaluate(
        harnesses, tasks, out, settings["base_url"], settings.get("model"), cache,
        concurrency=settings.get("concurrency", 8), timeout_sec=settings.get("timeout_sec"),
        api_key_env=settings.get("api_key_env", "OPENAI_API_KEY"), grade_answers=False, private=private,
    ))


def rollout_facts(folder: Path) -> Optional[dict]:
    """What a rollout did, label-free: steps, turns, when the answer was written, nudges, errors."""
    result_file, trajectory_file = folder / "result.json", folder / "trajectory.json"
    if not result_file.exists() or not trajectory_file.exists():
        return None
    result = json.loads(result_file.read_text())
    folder = Path(folder).resolve()
    conversation = json.loads(trajectory_file.read_text())["conversation"]
    steps, turn, answer_turn = [], 0, None
    results = {m.get("tool_call_id"): str(m.get("content", "")) for m in conversation if m["role"] == "tool"}
    for message in conversation:
        if message["role"] != "assistant":
            continue
        turn += 1
        for call in message.get("tool_calls") or []:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            # The rollout's own folder is long and the same in every call: show it as `.`
            code = ((arguments.get("code") or "") if name == "ipython" else json.dumps(arguments)).replace(str(folder), ".")
            output = results.get(call.get("id"), "").replace(str(folder), ".")
            error = TOOL_ERROR.search(output)
            if answer_turn is None and (name == "submit_answer" or ANSWER_WRITE.search(code or "")):
                answer_turn = turn
            steps.append({"turn": turn, "tool": name, "code": " ⏎ ".join((code or "").strip().splitlines())[:STEP_CHARS],
                          "output": " ⏎ ".join(output.strip().splitlines())[:STEP_CHARS],
                          "error": _error_line(output) if error else None})
    return {
        "task": result["task"], "turns": result["turns"], "timed_out": result["timed_out"],
        "tokens": result["prompt_tokens"] + result["completion_tokens"], "answer": result["answer"],
        "answer_turn": answer_turn, "after_answer": result["turns"] - answer_turn if answer_turn else None,
        "nudges": sum(m["role"] == "user" and m.get("content") == NUDGE for m in conversation),
        "errors": sum(step["error"] is not None for step in steps), "steps": steps,
    }


def _error_line(output: str) -> str:
    lines = [line for line in output.splitlines() if re.search(r"\w*(Error|Exception)\b", line)]
    return (lines[-1] if lines else output.strip().splitlines()[-1] if output.strip() else "")[:STEP_CHARS]


def digest(name: str, folder: Path, questions: Dict[str, str]) -> str:
    """Markdown digest of one harness's probe rollouts in `folder` (one subfolder per task)."""
    facts = [f for f in (rollout_facts(d) for d in sorted(Path(folder).iterdir()) if d.is_dir()) if f]
    if not facts:
        return f"# Probe: {name}\n\nNo finished rollouts.\n"
    after = [f["after_answer"] for f in facts if f["after_answer"] is not None]
    lines = [
        f"# Probe: {name} ({len(facts)} rollouts on train tasks, not graded)", "",
        f"- turns: mean {statistics.fmean(f['turns'] for f in facts):.1f}, median {statistics.median(f['turns'] for f in facts):.0f}, "
        f"max {max(f['turns'] for f in facts)}",
        f"- answer written in {len(after)}/{len(facts)} rollouts; turns spent after writing it: "
        f"mean {statistics.fmean(after) if after else 0:.1f}",
        f"- nudges (empty reply, no tool call): {sum(f['nudges'] for f in facts)}; timeouts: {sum(f['timed_out'] for f in facts)}",
        f"- tool calls with an error: {sum(f['errors'] for f in facts)} in {sum(len(f['steps']) for f in facts)}",
        f"- tokens per rollout: mean {statistics.fmean(f['tokens'] for f in facts) / 1e3:.1f}k", "",
    ]
    for f in facts:
        flags = [f"{f['turns']} turns", f"{f['tokens'] / 1e3:.1f}k tokens", f"answer {f['answer']!r}" if f["answer"] else "NO ANSWER"]
        if f["after_answer"]:
            flags.append(f"{f['after_answer']} turns after the answer")
        if f["nudges"]:
            flags.append(f"{f['nudges']} nudge(s)")
        if f["timed_out"]:
            flags.append("TIMED OUT")
        lines += [f"## {f['task']}: {questions.get(f['task'], '')}", "- " + ", ".join(flags)]
        for step in f["steps"]:
            lines.append(f"  - t{step['turn']} `{step['tool']}` {step['code']}")
            lines.append(f"    -> {'ERROR ' + step['error'] if step['error'] else step['output']}")
        lines.append("")
    return "\n".join(lines)


def contrast(folders: Dict[str, Path]) -> str:
    """Task-by-task table of the probe rollouts of several harnesses (the first is the reference)."""
    names = list(folders)
    facts = {name: {f["task"]: f for f in (rollout_facts(d) for d in sorted(Path(folder).iterdir()) if d.is_dir()) if f}
             for name, folder in folders.items()}
    tasks = sorted(set.intersection(*(set(f) for f in facts.values()))) if facts else []
    lines = [f"# Probe contrast: {', '.join(names)} ({len(tasks)} train tasks, not graded)", "",
             "Each cell: turns / errors / answer. `≠ref` marks an answer that differs from the reference's: "
             "a behaviour change worth reading in the digests, not a verdict on which is right.", "",
             "| task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for task in tasks:
        reference = facts[names[0]][task]["answer"].strip().lower()
        cells = []
        for name in names:
            f = facts[name][task]
            answer = f["answer"] or "NO ANSWER"
            mark = " ≠ref" if name != names[0] and f["answer"].strip().lower() != reference else ""
            cells.append(f"{f['turns']} / {f['errors']} / {answer[:30]!r}{mark}{' TIMEOUT' if f['timed_out'] else ''}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", "| totals | " + " | ".join(
        f"{statistics.fmean(f['turns'] for f in facts[n].values()):.1f} turns, {sum(f['errors'] for f in facts[n].values())} errors, "
        f"{sum(not f['answer'] for f in facts[n].values())} unanswered" for n in names) + " |"]
    return "\n".join(lines) + "\n"
