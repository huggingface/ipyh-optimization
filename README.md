# ipyh-optimization

Optimize an [ipython-harness](https://github.com/huggingface/ipython-harness) (ipyh) harness for a
suite of tasks, adapting [Harness²](https://github.com/google-research/harness2). There is no
training, and the agent is not run while optimizing. The result is an ipyh harness folder: swap it
in, and the same model should solve more tasks of the suite in fewer turns.

## How it works

Harness² improves an agent by editing its harness (prompt, tools, skills), not its weights, with a
Propose → Probe → Compose loop. `ipyh-optimize` keeps its structure and makes it cheap:

1. **Evidence**, the static stand-in for probe rollouts. The train tasks are inspected: what the
   questions ask, and each input file's encoding, delimiter, columns, quirks and sample rows. The
   optimizer is label-free: it never reads answers, graders or `tests/`.
2. **Propose.** Two scoped editors each edit a copy of the harness:
   - the INVARIANT editor works on agent mechanics (turns, output volume, error recovery, the
     finishing protocol);
   - the DOMAIN editor works on data handling and answer conventions.

   Each edit must cite its evidence in `CHANGELOG.md`.
3. **Validate.** Every candidate is loaded and its kernel tools are run in a fresh kernel. An
   invalid candidate gets one retry with the errors, then falls back to the base.
4. **Compose.** A composer merges the edits the evidence supports into one harness.
5. **Reflect.** Lessons go to a library (`EXPERIENCE.md`) that later rounds and runs read.

An editor is any coding agent. The default is Claude Code (`claude -p`), and any command works with
`--editor "<cmd> {instruction}"`. With `--editor none`, each stage stops for you, or for a Claude
Code session using the bundled `optimize-harness` skill, to do it by hand.

## Usage

```bash
pip install -e ".[eval]"   # installs ipyharness from GitHub

ipyh-optimize init runs/smol --tasks ~/SmolDataEnvs-harbor-train-sandbox --train 200 --heldout 100
ipyh-optimize step runs/smol             # -> runs/smol/harness
ipyh-optimize validate runs/smol/harness

# The only step that calls the model: base vs optimized on held-out tasks, graded by each task's grader
ipyh-optimize eval runs/smol --base-url http://localhost:8080/v1 --limit 50 --concurrency 4
ipyh-optimize report runs/smol           # pass rate, turns, paired 95% CIs, per difficulty
```

Use the optimized harness with ipyh directly:

```bash
python -m ipyh --harness-config runs/smol/harness --instruction-file task.md --output run.json
```

```py
from ipyh import Harness
agent = Harness.from_config("runs/smol/harness") + model
```

## Running the optimization step

`step` hands each stage (INV, DOM, compose, reflect) to an editor: a coding agent that reads the
stage's prompt file and edits a copy of the harness. Pick one of three ways to run it.

**Claude Code, unattended (default).** Install [Claude Code](https://claude.com/claude-code), log in
once, then:

```bash
ipyh-optimize step runs/smol
```

Each stage runs `claude -p` with file edits allowed only inside the run, the library and the input
cache. Logs go to `runs/smol/round-<k>/logs/`. A stage has 30 minutes by default (`--editor-timeout`).

**Any other coding agent.** Pass a command template. `{instruction}` is a one-line request telling
the agent which prompt file to follow:

```bash
ipyh-optimize step runs/smol --editor "opencode run {instruction}"
ipyh-optimize step runs/smol --editor "codex exec --full-auto {instruction}"
```

The template can also use `{prompt_file}`, `{workdir}`, `{run_dir}`, `{library}`, `{cache}` and
`{python}`. The agent needs to read and write files and run `python` (for `validate`).

**By hand, or in an interactive Claude Code session.** With `--editor none`, `step` stops at each
stage and prints the prompt file to follow:

```bash
ipyh-optimize step runs/smol --editor none   # repeat after each stage; exit code 3 means a stage is waiting
```

From a clone of this repo, Claude Code picks up the bundled `optimize-harness` skill, so you can just
ask it to "optimize the harness for runs/smol" and it plays every stage for you. The skill lives in
`.claude/skills/optimize-harness/`; copy it to `~/.claude/skills/` to use it from other projects.

Tasks are read in the [Harbor](https://github.com/laude-institute/harbor) format: `task.toml`,
`instruction.md`, `tests/`. Input data is pulled from the HF bucket named in the task's environment.

`eval` runs each task locally, without a container: the input files are linked into a folder, and the
instruction's paths are rewritten to point there. The model's code runs with your permissions, so
use a throwaway machine or container.

## Layout of a run

```
runs/<name>/
  config.json, splits.json      disjoint, stratified train / heldout splits
  base/                         the harness to improve
  round-<k>/evidence/           SUMMARY.md, BASE.md, tasks.jsonl, datasets/*.md
  round-<k>/inv/ dom/ final/    the two candidates and the composed harness
  round-<k>/prompts/ logs/      what each editor was asked, and its output
  harness/                      the result
  eval/                         rollouts, result.json per rollout, report.md
```

## Development

`ipyharness` is installed from GitHub. To work against a local checkout, install it after this
package, so it replaces the GitHub copy:

```bash
uv venv && uv pip install -e ".[dev,eval]" && uv pip install -e ../ipython-harness
pytest tests
```
