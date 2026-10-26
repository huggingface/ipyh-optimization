
## Probe results (read these before deciding)
Both candidates and the reference were run on the same train tasks (not graded):
`{{evidence}}/probe/CONTRAST.md` (task by task), with the digests `probe/inv.md` and `probe/dom.md`
and full trajectories under `{{round_dir}}/probe-runs/<name>/<task>/`.
- Adopt an edit only if its candidate's runs show the behaviour change the edit intends.
- Reject or adapt an edit whose runs show a regression:
  - new errors;
  - more turns or timeouts;
  - unanswered tasks;
  - answers that changed (`≠ref`) where the edit gives no reason for the change. Read those
    trajectories.
- A rule that the model misapplies in a probe run (e.g. applied to a question it does not fit)
  must be narrowed or dropped.
In `## Composer decisions`, cite the probe tasks behind each decision.
