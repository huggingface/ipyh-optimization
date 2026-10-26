# Optimizing an ipython-harness for a task suite

You are improving the **harness** of a coding agent: everything around a fixed model, i.e. its
system prompt, its kernel tools, its skills and its agent settings. The model is not changed and you
cannot run it. The goal is that the same model, given the improved harness, **solves more tasks of
this suite and uses fewer turns** (model calls).

You never see answers or graders, and you must not try to find them: improve the harness from
what the tasks ask, the data they ship with, and how the harness works. Do not open `tests/`,
`solution/` or `task.toml` files of the tasks.

## What you can read (read-only)
- `{{evidence}}/SUMMARY.md`: what the tasks ask and the pitfalls their data holds, with counts.
  Start here.
- `{{evidence}}/BASE.md`: the harness's mechanics and the current harness.
- `{{evidence}}/INSTRUCTION_EXAMPLE.md`: a task's instruction exactly as the agent receives it.
- `{{evidence}}/tasks.jsonl`: one card per task (question, answer kind, files, difficulty).
- `{{evidence}}/datasets/<name>.md`: each dataset's file profiles (encoding, delimiter, columns,
  quirks, sample rows) and the questions asked about it.
- `{{base}}/`: the harness you start from.
- `{{library}}/`: lessons and harnesses from earlier optimizations (`EXPERIENCE.md`, `INDEX.md`),
  if any. Reuse what fits the evidence here; do not copy blindly.

## The harness folder (your working directory: `{{workdir}}`)
```
harness.json          {"system_prompt_file": "system_prompt.md",
                       "tools": [{"name", "description", "parameters", "code_file"}],
                       "skills": ["skills"], "cell_timeout_sec": 300,
                       "agent": {"max_iteration": ..., "max_tokens": ...}}
system_prompt.md      must contain `{python}` once; any other brace is written `{{` / `}}`
tools/<name>.py       defines `def <name>(...)`; run in the agent's kernel at start
skills/<name>/SKILL.md  frontmatter `name`, `description` (when to use it), then instructions
CHANGELOG.md          one entry per edit (see below)
```
How the pieces reach the model:
- The system prompt is sent on **every** turn: keep it short and specific. Every sentence must
  earn its place.
- A kernel tool is offered to the model as its own function-calling tool (name, description, JSON
  schema). A call runs `name(**arguments)` as a cell: what it **prints** is the tool result. Print a
  compact result and return None; a returned string is shown as its repr. A tool must never
  raise on bad input: print a short, actionable message instead. Its code also defines a plain
  Python function the model can call from `ipython` cells.
- A skill costs one line (its description) in the system prompt; the model reads the SKILL.md with
  `view()` only when the description matches. Use skills for longer, situational know-how.
- `agent` settings: `max_iteration` caps turns per task, `max_tokens` caps one reply, and
  `stop_tools` lists kernel tools that end the run as soon as a call to them succeeds (output
  with no traceback; raise on bad input so the model can retry). Without one, the run ends
  only when the model replies with no tool call, which costs at least one more turn.

## Rules (from Harness²)
1. **Ground every edit in evidence.** Name the evidence file and the count or example that
   motivates it. Prefer issues that affect many tasks.
2. **Omitted is inherited.** The folder starts as a copy of the base: change only what your edit
   needs and leave the rest as it is.
3. **Small model, fixed budget.** Assume a mid-size open model with a limited turn and time budget.
   Defaults it can apply blindly beat options it has to reason about.
4. **Do not overfit.** No answers, task ids, or dataset-specific constants. Write rules and helpers
   that fit every task like these.
5. **Record each edit** in `CHANGELOG.md` as a bullet:
   `- [{{role}}] <component>: <what changed>. Why: <evidence file + count/example>.`
6. **Validate.** Run `{{validate}}` and fix every problem it reports, until it prints `OK`. You may
   also test tool code with plain Python on files under `{{cache}}`, the tasks' input files.
