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
"""Optimize an ipython-harness for a task suite, Harness²-style, without running the agent:

    ipyh-optimize init runs/opt --tasks ~/SmolDataEnvs-harbor-train-sandbox --train 200 --heldout 100
    ipyh-optimize step runs/opt                  # evidence -> INV + DOM editors -> composer -> reflect
    ipyh-optimize validate runs/opt/harness      # any harness folder
    ipyh-optimize eval runs/opt --base-url URL --model NAME --limit 50   # base vs optimized, held out
    ipyh-optimize report runs/opt

The result, `runs/opt/harness`, is a harness folder: `python -m ipyh --harness-config runs/opt/harness`
or `Harness.from_config("runs/opt/harness")`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ipyh_optimize.loop import DEFAULT_EDITOR, EDITOR_TIMEOUT_SEC, init_run, step
from ipyh_optimize.validate import validate_harness


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ipyh-optimize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("init", help="create a run: splits and base harness")
    p.add_argument("run", type=Path)
    p.add_argument("--tasks", type=Path, help="folder of Harbor tasks (or holding tasks/)")
    p.add_argument("--continue-from", type=Path,
                   help="an earlier run: keep its held-out tasks, draw unused train tasks, start from its harness")
    p.add_argument("--base", type=Path, help="harness folder or JSON to start from (default: ipyh's default)")
    p.add_argument("--train", type=int, default=200, help="tasks inspected to optimize")
    p.add_argument("--heldout", type=int, default=100, help="tasks kept for eval only")
    p.add_argument("--rounds", type=int, default=1, help="sequential rounds, each on its own train batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--library", type=Path, help="experience library shared across runs (default: <run>/library)")
    p.add_argument("--cache", type=Path, help="where task input files are downloaded")
    p.add_argument("--domain", help="library domain name (default: the tasks folder's name)")

    p = commands.add_parser("step", help="advance the optimization as far as possible")
    p.add_argument("run", type=Path)
    p.add_argument("--editor", default=DEFAULT_EDITOR,
                   help="editor command template ({instruction}, {prompt_file}, {workdir}, {run_dir}, "
                        "{library}, {cache}, {python}), or 'none' to do each stage by hand")
    p.add_argument("--editor-timeout", type=int, default=EDITOR_TIMEOUT_SEC)

    p = commands.add_parser("validate", help="check a harness folder")
    p.add_argument("harness", type=Path)
    p.add_argument("--no-smoke", action="store_true", help="skip the kernel smoke test")

    p = commands.add_parser("eval", help="run harnesses on held-out tasks (calls the model)")
    p.add_argument("run", type=Path)
    p.add_argument("--harness", nargs="+", metavar="NAME=PATH",
                   help="harnesses to compare (default: base=<run>/base optimized=<run>/harness)")
    p.add_argument("--split", default="heldout", choices=["heldout", "train"])
    p.add_argument("--limit", type=int, help="first N tasks of the split (it is stratified in order)")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-turns", type=int, help="override every harness's max_iteration")
    p.add_argument("--timeout-sec", type=float, help="per rollout (default: the task's agent timeout)")
    p.add_argument("--out", type=Path, help="default: <run>/eval")

    p = commands.add_parser("report", help="compare the eval results")
    p.add_argument("run", type=Path)
    p.add_argument("--eval-dir", type=Path, help="default: <run>/eval")

    args = parser.parse_args(argv)
    if args.command == "init":
        if not (args.tasks or args.continue_from):
            parser.error("init needs --tasks or --continue-from")
        root = init_run(args.run, args.tasks, args.base, args.train, args.heldout, args.rounds, args.seed,
                        args.library, args.cache, args.domain, args.continue_from)
        print(f"run created at {root}; next: ipyh-optimize step {args.run}")
    elif args.command == "step":
        finished = step(args.run, None if args.editor == "none" else args.editor, args.editor_timeout)
        sys.exit(0 if finished else 3)
    elif args.command == "validate":
        errors = validate_harness(args.harness, smoke=not args.no_smoke)
        print("OK" if not errors else "\n".join(f"- {e}" for e in errors))
        sys.exit(1 if errors else 0)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "report":
        write_report(args.run, args.eval_dir)


def run_eval(args) -> None:
    from ipyh_optimize.eval import evaluate
    from ipyh_optimize.tasks import load_tasks

    root = args.run.resolve()
    config = json.loads((root / "config.json").read_text())
    ids = json.loads((root / "splits.json").read_text())[args.split][: args.limit]
    pairs = args.harness or [f"base={root / 'base'}", f"optimized={root / 'harness'}"]
    harnesses = {name: Path(path) for name, path in (pair.split("=", 1) for pair in pairs)}
    for name, path in harnesses.items():
        errors = validate_harness(path, smoke=False)
        if errors:
            raise SystemExit(f"harness {name} ({path}) is not valid:\n" + "\n".join(errors))
    out = args.out or root / "eval"
    results = asyncio.run(evaluate(harnesses, load_tasks(Path(config["tasks"]), ids), out, args.base_url, args.model,
                                   Path(config["cache"]), args.concurrency, args.repeats, args.max_turns,
                                   args.timeout_sec, args.api_key_env))
    crashed = sum(bool(r.get("crashed")) for r in results)
    if crashed:
        print(f"{crashed} rollouts crashed before their first model call; see their agent.log, fix, and rerun "
              "(finished rollouts are kept)")
    write_report(root, out)


def write_report(root: Path, eval_dir: Path = None) -> None:
    from ipyh_optimize.report import load_results, render_report

    eval_dir = eval_dir or Path(root) / "eval"
    results = load_results(eval_dir)
    if not results:
        raise SystemExit(f"no results under {eval_dir}")
    report = render_report(results)
    (eval_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
