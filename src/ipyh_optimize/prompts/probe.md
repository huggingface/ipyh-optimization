- `{{evidence}}/probe/reference.md`: **how the current harness actually behaved** when run on a few
  train tasks with the target model (not graded: you see behaviour, never correctness). Per task:
  the tool sequence, errors, turns spent after the answer was written, nudges (empty replies),
  timeouts. Behaviour observed in runs outranks what static evidence suggests. Prefer edits that
  fix a pattern you can see there, and cite the task ids. Full trajectories:
  `{{round_dir}}/probe-runs/reference/<task>/trajectory.json`.
