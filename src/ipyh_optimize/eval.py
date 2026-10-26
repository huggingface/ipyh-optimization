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
"""Runs harnesses on tasks and grades them: the only step that calls the model, and it is not part
of the optimization (the editors never see its results).

Each rollout is `python -m ipyh` in its own folder, with no container: the task's input files are
linked into `input/`, the instruction's container paths are rewritten to the rollout's folders, and
the answer is graded with the task's own `tests/grader.py`. The model's code runs with your
permissions: use a throwaway machine or container.

    <out>/<harness>/<task>[-r<k>]/instruction.md, trajectory.json, agent.log, result.json
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ipyh_optimize.tasks import INPUT_DIR, WORK_DIR, Task, stage_inputs

ANSWER_FILE = "answer.txt"
SECRET_ENV = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL", re.I)  # names kept out of the agent's environment
TOOL_ERROR = re.compile(r"Traceback \(most recent call last\)|^\w*(Error|Exception)\b", re.M)


async def evaluate(harnesses: Dict[str, Path], tasks: List[Task], out: Path, base_url: str,
                   model: Optional[str], cache: Path, concurrency: int = 8, repeats: int = 1,
                   max_turns: Optional[int] = None, timeout_sec: Optional[float] = None,
                   api_key_env: str = "OPENAI_API_KEY", grade_answers: bool = True,
                   private: Iterable[Path] = ()) -> List[dict]:
    """Runs every harness on every task `repeats` times; finished rollouts are reused. With
    `grade_answers=False` (probe rollouts, inside the label-free loop) the reward is None."""
    out = Path(out).resolve()  # each rollout runs in its own folder, so every path it gets is absolute
    # What an agent must not read: other rollouts (their answers and grades), the task folders (their
    # task.toml holds the gold answer) and the Hub token. On macOS each rollout is sandboxed to enforce it
    private = [out, *private, *{task.path.parent for task in tasks}, Path.home() / ".cache" / "huggingface"]
    private = sorted({Path(p).resolve() for p in private})
    if not shutil.which("sandbox-exec"):
        print("WARNING: no sandbox-exec: rollouts are not isolated from each other or from the task folders; "
              "check results' `escape_attempts`, or run eval in a container", flush=True)
    print(f"staging inputs of {len(tasks)} tasks", flush=True)
    by_prefix = {task.bucket_prefix: task for task in tasks}
    with ThreadPoolExecutor(8) as pool:
        folders = dict(zip(by_prefix, pool.map(lambda t: stage_inputs(t, cache), by_prefix.values())))

    semaphore = asyncio.Semaphore(concurrency)
    # The agent's code runs with this environment: keep model and Hub credentials out of it. The
    # model key reaches `python -m ipyh` as OPENAI_API_KEY, which it removes before starting the kernel
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env["OPENAI_API_KEY"] = os.environ.get(api_key_env, "EMPTY")
    # Interleaved, so a partial evaluation still compares the harnesses on the same tasks
    jobs = [(name, path, task, k) for task in tasks for k in range(repeats) for name, path in harnesses.items()]
    folder_of = lambda name, task, k: out / name / (task.id if repeats == 1 else f"{task.id}-r{k}")
    done = sum((folder_of(name, task, k) / "result.json").exists() for name, _, task, k in jobs)
    if done:
        print(f"{done}/{len(jobs)} rollouts already finished: reusing them", flush=True)

    async def run(name: str, path: Path, task: Task, k: int) -> dict:
        nonlocal done
        folder = folder_of(name, task, k)
        result_file = folder / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        async with semaphore:
            if folder.exists():  # an interrupted attempt: start clean, so a stale answer is never graded
                shutil.rmtree(folder)
            result = await rollout(path, task, folder, folders[task.bucket_prefix], base_url, model, env,
                                   max_turns, timeout_sec or task.agent_timeout_sec, grade_answers, private)
        result.update(harness=name, repeat=k)
        done += 1
        if result["returncode"] and not result["timed_out"]:
            # `python -m ipyh` exits non-zero only when the run itself broke (model API, setup), never for
            # the agent's own errors: not a result. Nothing is saved, so the next `eval` retries it
            log = (folder / "agent.log").read_text(errors="replace").strip().splitlines()
            print(f"[{done}/{len(jobs)}] {name} {task.id}: CRASHED, not recorded: {log[-1] if log else '?'}", flush=True)
            return {**result, "crashed": True}
        result_file.write_text(json.dumps(result, indent=1))
        print(f"[{done}/{len(jobs)}] {name} {task.id}: reward={result['reward']} turns={result['turns']}"
              f"{' (timed out)' if result['timed_out'] else ''}", flush=True)
        return result

    return await asyncio.gather(*(run(*job) for job in jobs))


async def rollout(harness: Path, task: Task, folder: Path, inputs: Path, base_url: str, model: Optional[str],
                  env: dict, max_turns: Optional[int], timeout_sec: float, grade_answers: bool = True,
                  private: Iterable[Path] = ()) -> dict:
    input_dir, work_dir = folder / "input", folder / "workdir"
    for d in (input_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)
    for file in inputs.iterdir():
        link = input_dir / file.name
        if not file.name.startswith(".") and not link.exists():
            link.symlink_to(file.resolve())
    instruction = task.instruction.replace(INPUT_DIR, str(input_dir)).replace(WORK_DIR, str(work_dir))
    (folder / "instruction.md").write_text(instruction)

    command = [sys.executable, "-m", "ipyh", "--base-url", base_url, "--instruction-file", str(folder / "instruction.md"),
               "--output", str(folder / "trajectory.json"), "--harness-config", str(Path(harness).resolve())]
    if model:
        command += ["--model", model]
    if max_turns:
        command += ["--max-turns", str(max_turns)]
    allowed = [folder.resolve(), Path(harness).resolve()]
    if shutil.which("sandbox-exec"):
        (folder / "sandbox.sb").write_text(sandbox_profile(private, allowed))
        command = ["sandbox-exec", "-f", str(folder / "sandbox.sb"), *command]

    started, timed_out = time.monotonic(), False
    with (folder / "agent.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=work_dir, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout_sec)
        except asyncio.TimeoutError:
            timed_out = True
            # SIGINT first, so the run's `finally` shuts its kernel down; then kill what is left
            process.send_signal(signal.SIGINT)
            try:
                await asyncio.wait_for(process.wait(), 15)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

    answer_file = work_dir / ANSWER_FILE
    answer = answer_file.read_text(errors="replace").strip() if answer_file.exists() else ""
    if not grade_answers:
        reward = None
    else:
        reward = await asyncio.to_thread(grade, task, answer, folder / "grade.log") if answer else 0.0
    return {
        "task": task.id,
        "reward": reward,
        "answered": bool(answer),
        "answer": answer[:200],
        "timed_out": timed_out,
        "returncode": process.returncode,
        "seconds": round(time.monotonic() - started, 1),
        "difficulty": task.metadata.get("difficulty_tier"),
        "answer_kind": task.answer_kind,
        **trajectory_metrics(folder / "trajectory.json"),
        "escape_attempts": escape_attempts(folder / "trajectory.json", private, allowed),
    }


def grade(task: Task, answer: str, log: Path) -> float:
    """The task's own grader (`tests/grader.py`) on `answer`, with the verifier's environment."""
    env = {**os.environ, **{k: str(v) for k, v in task.config.get("verifier", {}).get("env", {}).items()}}
    with log.open("w") as f:
        result = subprocess.run([sys.executable, "-I", str(task.path / "tests" / "grader.py")], input=answer,
                                stdout=subprocess.PIPE, stderr=f, text=True, env=env, timeout=120)
    try:
        return float(result.stdout.split()[0])
    except (IndexError, ValueError):
        return 0.0


def trajectory_metrics(path: Path) -> dict:
    """Turns, tokens and tool use from a `python -m ipyh` output file."""
    if not path.exists():
        return {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0,
                "kernel_tool_calls": 0, "tool_errors": 0}
    record = json.loads(path.read_text())
    calls = [call for message in record["conversation"] for call in message.get("tool_calls") or []]
    results = [m for m in record["conversation"] if m["role"] == "tool"]
    return {
        "turns": len(record["turns"]),
        "prompt_tokens": sum(turn["prompt_tokens"] or 0 for turn in record["turns"]),
        "completion_tokens": sum(turn["completion_tokens"] or 0 for turn in record["turns"]),
        "tool_calls": len(calls),
        "kernel_tool_calls": sum(call.get("function", {}).get("name") != "ipython" for call in calls),
        "tool_errors": sum(bool(TOOL_ERROR.search(str(m.get("content", "")))) for m in results),
        "finished": record["done"],
    }


def sandbox_profile(private: Iterable[Path], allowed: Iterable[Path]) -> str:
    """A macOS sandbox profile: everything is allowed except reading or writing `private`, apart from
    the paths in `allowed` (the last matching rule wins)."""
    paths = lambda ps: " ".join(f'(subpath "{p}")' for p in ps)
    return (f"(version 1)\n(allow default)\n(deny file-read* file-write* {paths(private)})\n"
            f"(allow file-read* file-write* {paths(allowed)})\n")


def escape_attempts(path: Path, private: Iterable[Path], allowed: Iterable[Path]) -> int:
    """Tool calls that named a private path outside the rollout's own folder and harness, or were
    refused by the sandbox: an agent looking for answers elsewhere."""
    if not path.exists():
        return 0
    private, allowed = [str(p) for p in private], [str(p) for p in allowed]
    conversation = json.loads(path.read_text())["conversation"]
    refused = {m.get("tool_call_id") for m in conversation
               if m["role"] == "tool" and "Operation not permitted" in str(m.get("content", ""))}
    count = 0
    for message in conversation:
        for call in message.get("tool_calls") or []:
            named = re.findall(r"/[^\s'\"\\),;]+", call["function"].get("arguments") or "")
            outside = any(any(p.startswith(root) for root in private) and not any(p.startswith(a) for a in allowed)
                          for p in named)
            count += bool(outside or call.get("id") in refused)
    return count
