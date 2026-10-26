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
"""Stratified, disjoint splits: `train` tasks are inspected to optimize the harness, `heldout`
tasks are only used by `eval`."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from ipyh_optimize.tasks import Task


def stratified_order(tasks: List[Task], seed: int = 0) -> List[Task]:
    """`tasks` shuffled so that every prefix keeps the strata (difficulty × answer kind) in
    proportion: each task is placed at its rank within its stratum, as a fraction of the stratum."""
    rng = random.Random(seed)
    strata: Dict[str, List[Task]] = defaultdict(list)
    for task in tasks:
        strata[task.stratum()].append(task)
    keyed = []
    for members in strata.values():
        rng.shuffle(members)
        offset = rng.random()  # strata of the same size do not always come in the same order
        keyed += [((i + offset) / len(members), rng.random(), task) for i, task in enumerate(members)]
    return [task for *_, task in sorted(keyed, key=lambda key: key[:2])]


def make_splits(tasks: List[Task], n_train: int, n_heldout: int, rounds: int = 1, seed: int = 0,
                heldout: Optional[List[str]] = None, exclude: Iterable[str] = ()) -> dict:
    """{"heldout": ids, "train": ids, "rounds": [ids per round]}; the rounds partition `train`.

    A continued run passes the earlier `heldout` (kept as is, so every version is scored on the same
    tasks) and the earlier train tasks as `exclude`, so its train tasks are new."""
    order = [task.id for task in stratified_order(tasks, seed)]
    if heldout is None:
        heldout, order = order[:n_heldout], order[n_heldout:]
    unused = set(heldout) | set(exclude)
    order = [task_id for task_id in order if task_id not in unused]
    if n_train > len(order):
        raise ValueError(f"asked for {n_train} train tasks, {len(order)} are left")
    train = order[:n_train]
    return {
        "seed": seed,
        "heldout": heldout,
        "train": train,
        # Round-robin, so each round's batch is stratified too
        "rounds": [train[r::rounds] for r in range(rounds)],
    }
