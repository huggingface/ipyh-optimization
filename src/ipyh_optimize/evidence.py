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
"""The evidence the editors read, built without running the agent (the static stand-in for
Harness²'s probe rollouts):

    evidence/
      SUMMARY.md          what the tasks ask, and the pitfalls their data holds, with counts
      BASE.md             what the harness being optimized gives the agent, and its mechanics
      INSTRUCTION_EXAMPLE.md  one task's instruction, as the agent receives it
      tasks.jsonl         one label-free card per task (`Task.card()`)
      datasets/<name>.md  a profile of each input file, and the questions asked about it
"""

from __future__ import annotations

import codecs
import csv
import json
import re
import sqlite3
import statistics
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from ipyh.constants import CELL_TIMEOUT_SEC, MAX_OUTPUT_CHARS, MAX_TOKENS_PER_TURN
from ipyh.harness import Harness
from ipyh_optimize.tasks import Task, stage_inputs

PROFILE_ROWS = 50_000  # rows read to profile a table; the row count itself is exact
TABLE_EXTENSIONS = {".csv", ".tsv", ".txt", ".data", ".dat"}
# Files `profile_file` can read; the others (images, audio, documents) are listed, not downloaded
PROFILED_EXTENSIONS = TABLE_EXTENSIONS | {".xlsx", ".xls", ".json", ".parquet", ".sqlite", ".db", ".sqlite3"}
MISSING_TOKENS = {"?", "-", "--", "na", "n/a", "nan", "none", "null", "missing", "unknown", " "}
DATE_LIKE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
NUMBER_AS_TEXT = re.compile(r"^\s*[-+$€£]?\s*[\d,]+(\.\d+)?\s*%?\s*$")

# What a question asks for, matched on its words; a question can fall in several kinds
QUESTION_KINDS = {
    "count": r"\bhow many\b|\bnumber of\b|\bcount\b",
    "average": r"\baverage\b|\bmean\b|\bmedian\b",
    "extreme / ranking": r"\bhighest\b|\blowest\b|\bmost\b|\bleast\b|\bmax|\bmin|\btop\b|\blargest\b|\bsmallest\b",
    "percentage / ratio": r"percent|\bproportion\b|\bratio\b|\bshare\b|\brate\b|%",
    "group comparison": r"\bper\b|\bfor each\b|\bby\b|\bacross\b|\bgrouped\b",
    "missing values": r"\bmissing\b|\bnull\b|\bnan\b|\bimput",
    "unique / distinct": r"\bunique\b|\bdistinct\b",
    "correlation": r"\bcorrelat",
    "statistics test": r"p-value|\bt-test\b|\bchi|\banova\b|\bsignifican",
    "model / prediction": r"\bmodel\b|\bpredict|\baccuracy\b|\bregression\b|\btrain\b|\bclassif|\bf1\b|\brmse\b|\br2\b|r-squared|\bcluster",
    "date / time": r"\byear\b|\bmonth\b|\bdate\b|\bday\b|\bweek\b|\btime\b",
    "standard deviation / spread": r"\bstandard deviation\b|\bstd\b|\bvariance\b|\bquartile\b|\bpercentile\b|\biqr\b",
    "rounding asked": r"\bround|\bdecimal",
    "list answer": r"\bwhich\b.*\b(are|were)\b|\blist\b|\bname the\b",
}


def build_evidence(tasks: List[Task], harness_dir: Path, out: Path, cache: Path,
                   max_file_mb: float = 300, workers: int = 8) -> Path:
    """Writes `out/` for `tasks` and the harness at `harness_dir`; downloads inputs into `cache`."""
    out = Path(out)
    (out / "datasets").mkdir(parents=True, exist_ok=True)
    with (out / "tasks.jsonl").open("w") as f:
        for task in tasks:
            f.write(json.dumps(task.card()) + "\n")

    by_dataset: Dict[str, List[Task]] = defaultdict(list)
    for task in tasks:
        by_dataset[task.bucket_prefix.strip("/")].append(task)

    def profile_dataset(name: str) -> List[dict]:
        folder = stage_inputs(by_dataset[name][0], cache, max_bytes=int(max_file_mb * 2**20),
                              extensions=PROFILED_EXTENSIONS)
        files = sorted(path for path in folder.iterdir() if not path.name.startswith("."))
        profiles = [profile_file(path) for path in files if path.suffix.lower() in PROFILED_EXTENSIONS]
        others = defaultdict(list)  # extension -> sizes of the files that are not tables
        for path in files:
            if path.suffix.lower() not in PROFILED_EXTENSIONS:
                others[path.suffix.lower() or "none"].append(path.stat().st_size)
        for line in (folder / ".skipped").read_text().splitlines():
            file_name, size, reason = (line.split("\t") + ["too large"])[:3]
            if reason == "too large":
                profiles.append({"name": file_name, "size": int(size), "kind": "skipped",
                                 "quirks": [f"larger than {max_file_mb:.0f} MB: not downloaded for profiling"]})
            else:
                others[Path(file_name).suffix.lower() or "none"].append(int(size))
        for kind, sizes in sorted(others.items()):
            profiles.append({"name": f"{len(sizes)} {kind} file(s)", "size": sum(sizes), "kind": kind,
                             "count": len(sizes), "quirks": ["not a table: listed, not profiled"]})
        (out / "datasets" / f"{name}.md").write_text(render_dataset(name, profiles, by_dataset[name]))
        return profiles

    with ThreadPoolExecutor(workers) as pool:
        profiles = dict(zip(by_dataset, pool.map(profile_dataset, by_dataset)))

    (out / "INSTRUCTION_EXAMPLE.md").write_text(tasks[0].instruction)
    (out / "SUMMARY.md").write_text(render_summary(tasks, by_dataset, profiles))
    (out / "BASE.md").write_text(render_base(Harness.from_config(harness_dir)))
    return out


def profile_file(path: Path) -> dict:
    """Format, size, encoding, shape, column types and quirks of one input file; never raises."""
    import pandas as pd

    profile = {"name": path.name, "size": path.stat().st_size, "kind": path.suffix.lower() or "none", "quirks": []}
    quirks = profile["quirks"]
    try:
        suffix = path.suffix.lower()
        if suffix in TABLE_EXTENSIONS:
            raw = path.open("rb").read(1 << 16)
            encoding = "utf-8"
            if raw.startswith(codecs.BOM_UTF8):
                encoding = "utf-8-sig"
                quirks.append("UTF-8 byte-order mark: the first column name starts with \\ufeff unless read with encoding='utf-8-sig'")
            else:
                try:
                    codecs.getincrementaldecoder("utf-8")().decode(raw, final=False)
                except UnicodeDecodeError:
                    encoding = "latin-1"
                    quirks.append("not UTF-8: pd.read_csv fails without encoding='latin-1' (or 'cp1252')")
            text = raw.decode(encoding, errors="replace")
            try:
                delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
            if delimiter != ",":
                quirks.append(f"delimiter is not a comma: {delimiter!r}")
            profile.update(encoding=encoding, delimiter=delimiter)
            with path.open("rb") as f:
                profile["lines"] = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df = pd.read_csv(path, sep=delimiter, encoding=encoding, nrows=PROFILE_ROWS,
                                     low_memory=False, on_bad_lines="skip")
                except UnicodeDecodeError as error:  # valid UTF-8 at the start, not further on
                    quirks.append(f"not UTF-8: only past the first 64 KB ({error.reason} at byte {error.start}), "
                                  "so pd.read_csv fails without encoding='latin-1' (or 'cp1252')")
                    profile["encoding"] = "latin-1"
                    df = pd.read_csv(path, sep=delimiter, encoding="latin-1", nrows=PROFILE_ROWS,
                                     low_memory=False, on_bad_lines="skip")
            describe_frame(df, profile)
        elif suffix in (".xlsx", ".xls"):
            sheets = pd.read_excel(path, sheet_name=None, nrows=PROFILE_ROWS)
            profile["sheets"] = list(sheets)
            if len(sheets) > 1:
                quirks.append(f"several sheets: {len(sheets)}; pd.read_excel reads only the first unless sheet_name=None")
            describe_frame(next(iter(sheets.values())), profile)
        elif suffix == ".json":
            describe_frame(pd.read_json(path), profile)
        elif suffix in (".sqlite", ".db", ".sqlite3"):
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                profile["tables"] = [row[0] for row in db.execute("select name from sqlite_master where type='table'")]
        elif suffix == ".parquet":
            describe_frame(pd.read_parquet(path), profile)
    except Exception as error:  # the profile still says what failed, which is evidence too
        quirks.append(f"profiling failed: {type(error).__name__}: {str(error)[:200]}")
    return profile


def describe_frame(df, profile: dict) -> None:
    quirks = profile["quirks"]
    profile["rows_read"], profile["n_columns"] = df.shape
    profile["columns"] = [
        {"name": str(column), "dtype": str(df[column].dtype), "missing": round(float(df[column].isna().mean()), 3)}
        for column in df.columns[:60]
    ]
    profile["sample"] = df.head(3).to_csv(index=False)[:1500]
    names = [str(column) for column in df.columns]
    if any(name != name.strip() for name in names):
        quirks.append("column names with leading/trailing spaces")
    if any(name.startswith("Unnamed:") for name in names):
        quirks.append("a saved index or a blank header: columns named 'Unnamed N'")
    if any(re.search(r"\.\d+$", name) and name.rsplit(".", 1)[0] in names for name in names):
        quirks.append("duplicate column names: pandas renamed them with .1, .2")
    if df.shape[1] > 50:
        quirks.append(f"wide table: {df.shape[1]} columns, so df.head() output gets truncated")
    sparse = [str(column) for column in df.columns if df[column].isna().mean() > 0.2]
    if sparse:
        quirks.append(f"columns with more than 20% missing values: {', '.join(sparse[:10])}")
    for column in df.select_dtypes(include=["object", "string"]).columns[:60]:
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        head = values.head(200)
        if head.str.match(NUMBER_AS_TEXT).mean() > 0.9 and head.str.contains(r"[,$€£%]").any():
            quirks.append(f"numbers stored as text with separators or symbols: {column!r}, e.g. {head.iloc[0]!r}")
        elif head.str.match(DATE_LIKE).mean() > 0.9:
            quirks.append(f"dates stored as text: {column!r}, e.g. {head.iloc[0]!r}")
        if values.str.strip().str.lower().isin(MISSING_TOKENS).any():
            token = values[values.str.strip().str.lower().isin(MISSING_TOKENS)].iloc[0]
            quirks.append(f"missing values written as a token, not empty: {column!r} has {token!r}")
        unique = values.unique()
        if 1 < len(unique) <= 100 and len({value.strip().lower() for value in unique}) < len(unique):
            quirks.append(f"category spelled several ways (case or spaces): {column!r}")
        elif len(unique) > 1 and (values != values.str.strip()).mean() > 0.5:
            quirks.append(f"values with surrounding spaces: {column!r}")


def quirk_kind(quirk: str) -> str:
    """The quirk without its details (after the first colon), to count it across files."""
    return quirk.split(":", 1)[0].strip()


def render_dataset(name: str, profiles: List[dict], tasks: List[Task]) -> str:
    lines = [f"# Dataset `{name}`", "", f"{len(tasks)} task(s) use these files.", ""]
    for p in profiles:
        lines.append(f"## `{p['name']}` ({p['kind']}, {p['size'] / 2**20:.1f} MB)")
        for key in ("encoding", "delimiter", "lines", "rows_read", "n_columns", "sheets", "tables"):
            if key in p:
                lines.append(f"- {key}: {p[key]!r}")
        for quirk in p["quirks"]:
            lines.append(f"- QUIRK: {quirk}")
        if p.get("columns"):
            lines.append("- columns: " + ", ".join(
                f"{c['name']} ({c['dtype']}{', ' + format(c['missing'], '.0%') + ' missing' if c['missing'] else ''})"
                for c in p["columns"]))
        if p.get("sample"):
            lines += ["", "```csv", p["sample"].strip(), "```"]
        lines.append("")
    lines += ["## Questions", ""]
    lines += [f"- [{t.answer_kind}] {t.question} ({t.id})" for t in tasks]
    return "\n".join(lines) + "\n"


def render_summary(tasks: List[Task], by_dataset: Dict[str, List[Task]], profiles: Dict[str, List[dict]]) -> str:
    n = len(tasks)

    def table(counter: Counter, total: int, header: str, unit: str = "tasks") -> List[str]:
        rows = [f"| {header} | {unit} | share |", "|---|---|---|"]
        rows += [f"| {key} | {count} | {count / total:.0%} |" for key, count in counter.most_common()]
        return rows + [""]

    lines = [f"# Evidence summary ({n} tasks, {len(by_dataset)} datasets)", "",
             "Counts are over the inspected tasks. Each quirk names example datasets: read "
             "`datasets/<name>.md` for the file profile and the questions asked about it.", ""]
    lines += ["## Task mix", ""]
    lines += table(Counter(t.metadata.get("difficulty_tier", "?") for t in tasks), n, "difficulty")
    lines += table(Counter(t.answer_kind or "?" for t in tasks), n, "answer kind")

    kinds = Counter()
    examples = defaultdict(list)
    for task in tasks:
        for kind, pattern in QUESTION_KINDS.items():
            if re.search(pattern, task.question, re.I):
                kinds[kind] += 1
                if len(examples[kind]) < 3:
                    examples[kind].append(task.question)
    lines += ["## What the questions ask (keyword match; a question can count several times)", ""]
    for kind, count in kinds.most_common():
        lines.append(f"- **{kind}**: {count} ({count / n:.0%}); e.g. " + " | ".join(f"\"{q}\"" for q in examples[kind]))
    lines.append("")

    files = [p for dataset in profiles.values() for p in dataset]
    sizes = [p["size"] / 2**20 for p in files] or [0]
    files_per_task = [len(t.files) for t in tasks] or [0]
    lines += ["## Input files", "",
              f"- {sum(p.get('count', 1) for p in files)} files; size median {statistics.median(sizes):.1f} MB, max {max(sizes):.0f} MB; "
              f"{sum(s > 100 for s in sizes)} over 100 MB",
              f"- files per task: median {statistics.median(files_per_task):.0f}, max {max(files_per_task)}; "
              f"{sum(f > 1 for f in files_per_task)} tasks have several files", ""]
    kinds = Counter()
    for p in files:
        kinds[p["kind"]] += p.get("count", 1)
    lines += table(kinds, sum(kinds.values()) or 1, "file type", "files")

    tasks_hit, datasets_hit = Counter(), defaultdict(list)
    for name, dataset in profiles.items():
        for kind in {quirk_kind(q) for p in dataset for q in p["quirks"]}:
            tasks_hit[kind] += len(by_dataset[name])
            datasets_hit[kind].append(name)
    lines += ["## Data quirks (tasks whose data has the quirk)", ""]
    for kind, count in tasks_hit.most_common():
        lines.append(f"- {kind}: {count} tasks ({count / n:.0%}), {len(datasets_hit[kind])} datasets, "
                     f"e.g. {', '.join(datasets_hit[kind][:3])}")
    return "\n".join(lines) + "\n"


def render_base(harness: Harness) -> str:
    tools = "\n".join(f"- `{t['name']}`: {t['description']}" for t in harness.kernel_tools.values()) or "- none"
    skills = "\n".join(f"- `{s.name}`: {s.description}" for s in harness.skills) or "- none"
    return f"""# The harness being optimized

## Mechanics (ipython-harness, not editable through the harness folder)
- One model call is one turn; every tool call in a reply runs in order in one persistent IPython kernel.
- Tools: `ipython` (runs a cell), plus each kernel tool below (a Python function defined in the
  kernel at start; a call `f(**args)` runs as a cell, so its printed output is the tool result).
- Kernel helpers always defined: `view(path, start, n)` and `edit(path, old, new)`.
- A cell is interrupted after {harness.cell_timeout_sec or CELL_TIMEOUT_SEC} s (`# timeout: N` first line overrides).
- Tool output over {MAX_OUTPUT_CHARS} characters keeps its head and tail; the full output is saved to a file.
- At most {harness.agent_settings.get('max_tokens', MAX_TOKENS_PER_TURN)} tokens per reply (reasoning + answer).
- The run ends when the model replies with text and no tool call (one nudge is sent for an empty reply),
  or after `max_iteration` turns ({harness.agent_settings.get('max_iteration', 'no limit')} here), or right after a
  successful call to a tool in `agent.stop_tools` ({harness.agent_settings.get('stop_tools') or 'none'} here).
- Skills: their names and descriptions are listed in the system prompt; the model reads a SKILL.md with `view()`.

## Agent settings
{json.dumps(harness.agent_settings or {})}

## Kernel tools
{tools}

## Skills
{skills}

## System prompt (`{{python}}` is filled with a sentence about the interpreter)
```
{harness.system_prompt}
```
"""
