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
"""The optimization loop, Harness²'s Propose → Probe → Compose without rollouts.

A run folder holds everything, so each stage can be redone or done by hand:

    <run>/config.json, splits.json
    <run>/base/                       the harness to improve
    <run>/round-<k>/evidence/         static evidence from round k's train batch (see evidence.py)
    <run>/round-<k>/inv/, dom/        the two scoped candidates (copies of the round's base, edited)
    <run>/round-<k>/diffs/            each candidate against the base, for the composer
    <run>/round-<k>/final/            the composed harness; the next round's base
    <run>/round-<k>/prompts/, logs/   what each editor was asked, and its output
    <run>/harness/                    the result: the last round's final harness
    <library>/                        EXPERIENCE.md, INDEX.md, harnesses/ (shared across runs)

An editor is any command that edits files: by default Claude Code in print mode. With no editor
(`--editor none`), each stage stops after writing its prompt so a person, or a Claude Code session,
can do it; running `step` again validates that stage and goes on.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from ipyh.harness import HARNESS_FILE, Harness
from ipyh_optimize.evidence import build_evidence
from ipyh_optimize.probe import contrast, digest, run_probe
from ipyh_optimize.select import make_splits
from ipyh_optimize.tasks import load_tasks
from ipyh_optimize.validate import validate_harness

PROMPTS = Path(__file__).parent / "prompts"
STAGES = ("inv", "dom", "compose", "reflect")
# With probing: the round's base is run before the editors, and both candidates before the composer
PROBE_STAGES = ("reference", "inv", "dom", "probe", "compose", "reflect")
DEFAULT_CACHE = Path("~/.cache/ipyh-optimize/inputs").expanduser()
# `{instruction}` points the editor at its prompt file; the other fields are paths
DEFAULT_EDITOR = (
    "claude -p {instruction} --permission-mode acceptEdits --add-dir {run_dir} --add-dir {library} "
    "--add-dir {cache} --allowedTools Read Write Edit Glob Grep 'Bash({python}:*)' 'Bash(diff:*)' 'Bash(ls:*)'"
)
EDITOR_TIMEOUT_SEC = 1800


def init_run(root: Path, tasks_dir: Optional[Path] = None, base: Optional[Path] = None, n_train: int = 200,
             n_heldout: int = 100, rounds: int = 1, seed: int = 0, library: Optional[Path] = None,
             cache: Optional[Path] = None, domain: Optional[str] = None,
             continue_from: Optional[Path] = None, splits_from: Optional[Path] = None,
             probe: Optional[dict] = None) -> Path:
    """Creates the run folder: the splits, and the base harness (ipyh's default when `base` is None).

    With `continue_from` (an earlier run), the new run keeps that run's held-out tasks, draws train
    tasks it has not used, and by default starts from its harness, library, tasks and cache."""
    root = Path(root).resolve()
    if (root / "config.json").exists():
        raise FileExistsError(f"{root} already holds a run")
    previous, heldout, exclude = {}, None, ()
    if continue_from:
        continue_from = Path(continue_from).resolve()
        previous = json.loads((continue_from / "config.json").read_text())
        earlier = json.loads((continue_from / "splits.json").read_text())
        heldout, exclude = earlier["heldout"], earlier["train"] + earlier.get("excluded", [])
        base = base or continue_from / "harness"
    if splits_from:  # the exact splits of another run, for a controlled comparison
        splits_from = Path(splits_from).resolve()
        previous = previous or json.loads((splits_from / "config.json").read_text())
    tasks_dir = Path(tasks_dir or previous["tasks"]).expanduser().resolve()
    if splits_from:
        splits = json.loads((splits_from / "splits.json").read_text())
        rounds = len(splits["rounds"])
    else:
        splits = make_splits(load_tasks(tasks_dir), n_train, n_heldout, rounds, seed, heldout=heldout, exclude=exclude)
    if continue_from:
        splits["excluded"] = list(exclude)  # every train task of the earlier runs, for the next one
    root.mkdir(parents=True, exist_ok=True)
    (Harness.from_config(base) if base else Harness()).save(root / "base")
    if base and (Path(base) / "CHANGELOG.md").is_file():  # the editors read how the base came to be
        shutil.copy(Path(base) / "CHANGELOG.md", root / "base" / "CHANGELOG.md")
    _write_json(root / "splits.json", splits)
    library = library or previous.get("library")
    cache = cache or previous.get("cache")
    _write_json(root / "config.json", {
        "tasks": str(tasks_dir),
        "rounds": rounds,
        "domain": domain or previous.get("domain") or tasks_dir.name,
        "library": str(Path(library).expanduser().resolve() if library else root / "library"),
        "cache": str(Path(cache).expanduser().resolve() if cache else DEFAULT_CACHE),
        **({"continued_from": str(continue_from)} if continue_from else {}),
        **({"splits_from": str(splits_from)} if splits_from else {}),
        **({"probe": probe} if probe else {}),
    })
    return root


def step(root: Path, editor: Optional[str] = DEFAULT_EDITOR, editor_timeout: int = EDITOR_TIMEOUT_SEC) -> bool:
    """Advances the run as far as it can; True once `<run>/harness` is written. With `editor=None`
    it stops at the first stage that waits for a manual edit."""
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    library, cache = Path(config["library"]), Path(config["cache"])
    for folder in (library, cache):  # editors are given read access to both
        folder.mkdir(parents=True, exist_ok=True)

    for k in range(1, config["rounds"] + 1):
        round_dir = root / f"round-{k}"
        base = root / "base" if k == 1 else root / f"round-{k - 1}" / "final"
        state_file = round_dir / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get("evidence") != "done":
            tasks = load_tasks(Path(config["tasks"]), splits["rounds"][k - 1])
            print(f"[round {k}] inspecting {len(tasks)} tasks", flush=True)
            build_evidence(tasks, base, round_dir / "evidence", cache)
            state["evidence"] = "done"
            _write_json(state_file, state)

        paths = {
            "run_dir": root, "round_dir": round_dir, "base": base, "evidence": round_dir / "evidence",
            "library": library, "cache": cache, "inv": round_dir / "inv", "dom": round_dir / "dom",
            "final": round_dir / "final", "diffs": round_dir / "diffs",
        }
        context = {**{key: str(value) for key, value in paths.items()},
                   "domain": config["domain"], "entry": f"{root.name}-round-{k}", "round": str(k)}
        for stage in PROBE_STAGES if config.get("probe") else STAGES:
            if state.get(stage) == "done":
                continue
            if stage in ("reference", "probe"):
                probe_stage(stage, paths, config, splits["rounds"][k - 1])
            elif not run_stage(stage, context, state.get(stage), editor, editor_timeout):
                state[stage] = "pending"
                _write_json(state_file, state)
                return False
            state[stage] = "done"
            _write_json(state_file, state)

    last = root / f"round-{config['rounds']}" / "final"
    out = root / "harness"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(last, out)
    harness_config = json.loads((out / HARNESS_FILE).read_text())
    harness_config["meta"] = {
        "optimized_by": "ipyh-optimize",
        "tasks": config["tasks"],
        "rounds": config["rounds"],
        "train_tasks": len(splits["train"]),
    }
    _write_json(out / HARNESS_FILE, harness_config)
    print(f"optimized harness: {out}")
    return True


def run_stage(stage: str, context: Dict[str, str], status: Optional[str], editor: Optional[str],
              editor_timeout: int) -> bool:
    """Sets up and runs one stage; False when it waits for a manual edit."""
    workdir = Path({"inv": context["inv"], "dom": context["dom"], "compose": context["final"],
                    "reflect": context["library"]}[stage])
    round_dir = Path(context["round_dir"])
    if status is None:
        setup_stage(stage, workdir, context)
    prompt_file = round_dir / "prompts" / f"{stage}.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(render_prompt(stage, workdir, context))

    if editor is None:
        if status != "pending":
            print(f"\n[manual] {stage}: follow {prompt_file}\n         working directory: {workdir}\n"
                  "         then run `step` again to validate it and continue.")
            return False
        errors = check_stage(stage, workdir)
        if errors:
            print(f"[manual] {stage} is not valid yet:\n" + "\n".join(f"- {e}" for e in errors))
            return False
        return True

    log = round_dir / "logs" / f"{stage}.log"
    run_editor(editor, prompt_file, workdir, context, log, editor_timeout)
    errors = check_stage(stage, workdir)
    if errors:  # one retry with the validator's messages, as in Harness²
        prompt_file.write_text(render_prompt(stage, workdir, context, errors))
        run_editor(editor, prompt_file, workdir, context, log, editor_timeout)
        errors = check_stage(stage, workdir)
    if errors and stage != "reflect":
        print(f"[{stage}] still invalid after a retry; falling back to the base harness", flush=True)
        shutil.rmtree(workdir)
        shutil.copytree(context["base"], workdir)
        with (workdir / "CHANGELOG.md").open("a") as f:
            f.write(f"\n- [{stage}] fell back to the base harness; validation errors: {'; '.join(errors)[:1000]}\n")
    return True


def setup_stage(stage: str, workdir: Path, context: Dict[str, str]) -> None:
    if stage == "reflect":
        workdir.mkdir(parents=True, exist_ok=True)
        for name, header in (("EXPERIENCE.md", "# Experience"), ("INDEX.md", "# Harnesses")):
            if not (workdir / name).exists():
                (workdir / name).write_text(f"{header} ({context['domain']})\n")
        entry = workdir / "harnesses" / context["entry"]
        if entry.exists():
            shutil.rmtree(entry)
        shutil.copytree(context["final"], entry)
        return
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(context["base"], workdir)
    (workdir / "CHANGELOG.md").write_text(f"# Changes in round {context['round']}\n\n")
    if stage == "compose":
        diffs = Path(context["diffs"])
        diffs.mkdir(parents=True, exist_ok=True)
        for role in ("inv", "dom"):
            result = subprocess.run(["diff", "-ruN", context["base"], context[role]], capture_output=True, text=True)
            (diffs / f"{role}.diff").write_text(result.stdout)


def probe_stage(stage: str, paths: Dict[str, Path], config: dict, batch: List[str]) -> None:
    """Runs the probe rollouts (never graded) and writes their digests into the round's evidence."""
    settings = config["probe"]
    tasks = load_tasks(Path(config["tasks"]), batch[: settings["tasks"]])
    runs, out = paths["round_dir"] / "probe-runs", paths["evidence"] / "probe"
    out.mkdir(parents=True, exist_ok=True)
    harnesses = {"reference": paths["base"]} if stage == "reference" else {
        "reference": paths["base"], "inv": paths["inv"], "dom": paths["dom"]}  # reference rollouts are reused
    print(f"[{stage}] probing {', '.join(harnesses)} on {len(tasks)} train tasks", flush=True)
    # The folder holding every run is off limits to the agent (other rollouts, evals, evidence)
    results = run_probe(harnesses, tasks, runs, Path(config["cache"]), settings, private=[paths["run_dir"].parent])
    crashed = sum(bool(r.get("crashed")) for r in results)
    if crashed > len(results) // 2:  # e.g. a missing API key: do not hand the editors an empty probe
        raise SystemExit(f"[{stage}] {crashed}/{len(results)} probe rollouts crashed; see their agent.log under "
                         f"{runs}, fix (model URL, API key in ${settings.get('api_key_env')}) and run `step` again")
    questions = {task.id: task.question for task in tasks}
    for name in harnesses:
        (out / f"{name}.md").write_text(digest(name, runs / name, questions))
    if stage == "probe":
        (out / "CONTRAST.md").write_text(contrast({name: runs / name for name in harnesses}))


def render_prompt(stage: str, workdir: Path, context: Dict[str, str], errors: Optional[List[str]] = None) -> str:
    values = {
        **context,
        "workdir": str(workdir),
        "role": stage.upper(),
        "validate": f"{shlex.quote(sys.executable)} -m ipyh_optimize validate {shlex.quote(str(workdir))}",
        "errors": "" if not errors else (
            "\n## Fix these first\nYour previous edit did not validate:\n" + "\n".join(f"- {e}" for e in errors)
        ),
    }
    probe = Path(context["evidence"]) / "probe"
    values["probe"] = _fill((PROMPTS / "probe.md").read_text(), values) if (probe / "reference.md").exists() else ""
    values["probe_compose"] = (_fill((PROMPTS / "probe_compose.md").read_text(), values)
                               if (probe / "CONTRAST.md").exists() else "")
    if stage != "reflect":
        values["common"] = _fill((PROMPTS / "common.md").read_text(), values)
    return _fill((PROMPTS / f"{stage}.md").read_text(), values)


def check_stage(stage: str, workdir: Path) -> List[str]:
    if stage == "reflect":
        lessons = (workdir / "EXPERIENCE.md").read_text()
        return [] if "Lesson" in lessons else ["EXPERIENCE.md has no lesson (`Lesson:` lines)"]
    return validate_harness(workdir)


def run_editor(template: str, prompt_file: Path, workdir: Path, context: Dict[str, str], log: Path,
               timeout: int) -> None:
    values = {
        **context,
        "workdir": str(workdir),
        "prompt_file": str(prompt_file),
        "python": sys.executable,
        "instruction": f"Read {prompt_file} and do exactly what it says. Your working directory is {workdir}.",
    }
    # Each token is filled after splitting, so a value with spaces stays one argument
    command = [_fill(token, values, braces="{}") for token in shlex.split(template)]
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[{prompt_file.stem}] editing {workdir}", flush=True)
    with log.open("a") as f:
        f.write(f"$ {shlex.join(command)}\n")
        f.flush()
        try:
            subprocess.run(command, cwd=workdir, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            f.write(f"\n[editor timed out after {timeout}s]\n")
        except FileNotFoundError as error:
            raise SystemExit(f"editor command not found: {error}; pass --editor") from error
    print(f"[{prompt_file.stem}] done in {time.monotonic() - started:.0f}s (log: {log})", flush=True)


def _fill(text: str, values: Dict[str, str], braces: str = "{{}}") -> str:
    """`text` with each `{{key}}` (or `{key}`) of `values` replaced; other braces are kept."""
    left, right = braces[: len(braces) // 2], braces[len(braces) // 2:]
    pattern = re.escape(left) + r"(\w+)" + re.escape(right)
    return re.sub(pattern, lambda m: values.get(m.group(1), m.group(0)), text)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
