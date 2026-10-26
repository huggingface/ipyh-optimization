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
"""Harbor tasks (`task.toml`, `instruction.md`, `tests/`) and their input data.

The optimizer is label-free, as in Harness²: what it shows the editors comes from `Task.card()`,
which keeps an allowlist of fields. Gold answers, the verifier's environment and `tests/` are
only read by `eval`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

# `[metadata]` fields an editor may see: what kind of task it is, never its answer
CARD_METADATA = ("difficulty", "difficulty_tier", "difficulty_level", "package_tier", "kaggle_dataset_name")
# Paths the instructions use, rewritten when a task runs outside its container
INPUT_DIR, WORK_DIR = "/home/user/input", "/workdir"
_FILE_LINE = re.compile(r"^\s*-\s+(\S.*?)\s*$")


@dataclass
class Task:
    id: str
    path: Path
    config: dict = field(repr=False)

    @property
    def instruction(self) -> str:
        return (self.path / "instruction.md").read_text()

    @property
    def question(self) -> str:
        return self.config.get("task", {}).get("description", "").strip()

    @property
    def metadata(self) -> dict:
        return self.config.get("metadata", {})

    @property
    def answer_kind(self) -> str:
        """The answer's format (numeric, list, exact_short, …): a format contract, not a label."""
        return self.metadata.get("reward_mode_initial", "")

    @property
    def bucket(self) -> Optional[str]:
        return self.config.get("environment", {}).get("env", {}).get("HF_BUCKET")

    @property
    def bucket_prefix(self) -> Optional[str]:
        prefix = self.config.get("environment", {}).get("env", {}).get("BUCKET_PREFIX")
        return prefix.rstrip("/") + "/" if prefix else None

    @property
    def files(self) -> List[str]:
        """The input files the instruction lists under its `Files (...)` line."""
        names, listing = [], False
        for line in self.instruction.splitlines():
            if line.startswith("Files"):
                listing = True
            elif listing:
                match = _FILE_LINE.match(line)
                if not match:
                    break
                names.append(match.group(1))
        return names

    @property
    def agent_timeout_sec(self) -> float:
        return float(self.config.get("agent", {}).get("timeout_sec", 600))

    def card(self) -> Dict:
        """What the optimizer may know about the task (label-free)."""
        return {
            "id": self.id,
            "question": self.question,
            "answer_kind": self.answer_kind,
            "files": self.files,
            **{key: self.metadata[key] for key in CARD_METADATA if key in self.metadata},
        }

    def stratum(self) -> str:
        return f"{self.metadata.get('difficulty_tier', '?')}/{self.answer_kind or '?'}"


def load_task(path: Path) -> Task:
    path = Path(path)
    config = tomllib.loads((path / "task.toml").read_text())
    return Task(id=path.name, path=path, config=config)


def load_tasks(root: Path, ids: Optional[List[str]] = None) -> List[Task]:
    """The tasks under `root` (a folder of task folders, or one with a `tasks/` folder)."""
    root = Path(root).expanduser()
    if (root / "tasks").is_dir():
        root = root / "tasks"
    if ids is not None:
        return [load_task(root / task_id) for task_id in ids]
    return [load_task(p.parent) for p in sorted(root.glob("*/task.toml"))]


def stage_inputs(task: Task, cache: Path, max_bytes: Optional[int] = None,
                 extensions: Optional[Iterable[str]] = None) -> Path:
    """Downloads the task's input files from its HF bucket into `cache/<prefix>` (shared by the
    tasks on one dataset) and returns that folder. Files over `max_bytes`, or whose extension is not
    in `extensions`, are skipped and listed in `.skipped` (name, size, reason)."""
    from huggingface_hub import download_bucket_files, list_bucket_tree

    folder = Path(cache) / task.bucket_prefix.strip("/")
    complete = folder / ".complete"  # every file is there; `.skipped` lists the ones left out
    if complete.exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    targets, skipped = [], []
    for item in list_bucket_tree(task.bucket, prefix=task.bucket_prefix, recursive=True):
        if getattr(item, "type", None) != "file":
            continue
        local = folder / Path(item.path).name
        if local.exists() and local.stat().st_size == item.size:
            continue
        if extensions is not None and local.suffix.lower() not in extensions:
            skipped.append(f"{local.name}\t{item.size}\tnot profiled")
            continue
        if max_bytes and item.size > max_bytes:
            skipped.append(f"{local.name}\t{item.size}\ttoo large")
            continue
        targets.append((item.path, str(local)))
    if targets:
        download_bucket_files(task.bucket, files=targets)
    (folder / ".skipped").write_text("\n".join(skipped))
    if not skipped:
        complete.write_text("")
    return folder
