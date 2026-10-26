# Reflect: keep what this optimization round learned

You maintain a library of harness-optimization experience for the domain **{{domain}}**. Your
working directory is the library: `{{library}}`. It holds:
- `EXPERIENCE.md`: lessons, one per `##` section.
- `INDEX.md`: one line per stored harness and what it targets.
- `harnesses/`: the harness trees adopted so far.

This round's final harness is at `{{final}}` (stored as `harnesses/{{entry}}`, already copied).
Read its `CHANGELOG.md`, especially the composer decisions. Also read the evidence it was built
from, in `{{evidence}}` (start with `SUMMARY.md`).

Update the library:
1. **`EXPERIENCE.md`.** Add or update lessons that will transfer to future suites of this domain.
   Each lesson has three lines: `Cue:` (what in the evidence signals the situation), `Lesson:`
   (the harness change that addresses it) and `Evidence:` (counts from this round). Merge a lesson
   that already exists instead of repeating it. Note when this round contradicts an earlier
   lesson. These lessons come from static evidence and have not been verified by runs: say so,
   and never state that something improved the score.
2. **`INDEX.md`.** Add one line: `harnesses/{{entry}} — <what it targets, in a few words>`.
Do not edit anything else.
