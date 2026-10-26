{{common}}

## Your role: COMPOSER
Two editors each proposed a candidate harness from the same base:
- `{{inv}}/`: the INVARIANT editor (agent mechanics, turn efficiency, finishing protocol)
- `{{dom}}/`: the DOMAIN editor (data handling, analytic conventions, answer format)
Each has a `CHANGELOG.md` with its reasons. The diffs against the base are in `{{diffs}}/`.

Your working directory starts as a copy of the base. Build the final harness there. For each
proposed edit, decide whether to **adopt**, **adapt** or **reject** it:
- Adopt edits that the evidence supports, and check the evidence they cite yourself.
- Merge overlaps into one. Two loaders, or two answer-writing tools, become one; the system
  prompt must stay one coherent, short text and never state a rule twice.
- Reject edits that are speculative, contradict each other or the task instructions, or overfit
  to a dataset. Also reject any that inflate the per-turn prompt without a frequent payoff.
- Keeping the base, or adopting one candidate wholesale, is also a valid outcome.

In `CHANGELOG.md` keep the adopted entries, and add a `## Composer decisions` section with one
line per proposed edit: `adopted | adapted | rejected — reason`.
{{probe_compose}}
{{errors}}
