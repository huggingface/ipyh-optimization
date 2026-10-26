---
name: optimize-harness
description: Optimize an ipython-harness (ipyh) harness for a suite of tasks (e.g. Harbor tasks) with ipyh-optimize, Harness²-style and without rollouts, playing the INV editor, DOM editor, composer and reflector roles in this session. Use when asked to optimize, tune or improve a harness for a set of tasks or a benchmark, or to build an ipyh harness folder for one.
---

# Optimize an ipyh harness for a task suite

`ipyh-optimize` (`python -m ipyh_optimize`, from the ipyh-optimization package) builds evidence from the tasks themselves (no agent
runs, no answers), then a harness is improved in four stages per round: an INVARIANT editor
(agent mechanics), a DOMAIN editor (data and answer conventions), a composer that merges the two,
and a reflector that writes lessons to a library. You play each stage here; the tool prepares,
validates and records them.

## Steps
1. Create the run (pick sizes with the user; evidence needs the tasks' input files, downloaded
   once into a cache):
   `python -m ipyh_optimize init runs/<name> --tasks <tasks dir> --train 200 --heldout 100 [--rounds 1] [--base <harness>] [--library <shared library dir>]`
2. Run `python -m ipyh_optimize step runs/<name> --editor none`. It builds the round's evidence,
   then stops at the next stage and prints the prompt file and the working directory.
3. Read that prompt file **in full** and do exactly what it says, editing only files in the
   working directory it names. Respect the rules there: label-free (never open the tasks'
   `task.toml`, `tests/` or `solution/`), ground each edit in the evidence, and log it in
   CHANGELOG.md. Run the `validate` command it gives until it prints `OK`.
4. Run `step ... --editor none` again: it validates the stage and moves on to the next one
   (inv → dom → compose → reflect, then the next round). Repeat 3–4 until it prints
   `optimized harness: runs/<name>/harness` (exit code 0; 3 means a stage is waiting).
5. Keep the roles separate. While playing DOM, do not read the INV candidate. The composer is
   the one stage that reads both candidates and the diffs.
6. Summarize for the user: the main edits and their evidence (from `harness/CHANGELOG.md`), and
   how to measure them. Measuring calls a model, so ask before running it:
   `python -m ipyh_optimize eval runs/<name> --base-url <OpenAI-compatible URL> --model <name> --limit 50`
   then `python -m ipyh_optimize report runs/<name>`.

To run the editors unattended instead, omit `--editor none`: the default editor is
`claude -p` (Claude Code in print mode). Any command template works, e.g.
`--editor "opencode run {instruction}"`.
