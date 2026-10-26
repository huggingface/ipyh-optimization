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
"""Checks a harness folder before it is used: Harness² validates every proposed tree and gives
the editor one retry with the errors. No model is called; the smoke test only starts a kernel."""

from __future__ import annotations

import json
import re
import subprocess
import warnings
from pathlib import Path
from typing import List

from ipyh.harness import HARNESS_FILE, Harness

RESERVED_NAMES = {"ipython", "view", "edit"}
MAX_TOOLS = 12
MAX_CODE_CHARS = 20_000
MAX_PROMPT_CHARS = 12_000
MAX_SKILLS = 12


def validate_harness(folder: Path, smoke: bool = True) -> List[str]:
    """The problems found in the harness at `folder`, as messages for an editor ([] when valid)."""
    folder = Path(folder)
    if not (folder / HARNESS_FILE).is_file():
        return [f"{folder / HARNESS_FILE} is missing"]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            harness = Harness.from_config(folder)
    except Exception as error:
        return [f"Harness.from_config failed: {type(error).__name__}: {error}"]

    errors = [f"skill warning: {w.message}" for w in caught]
    try:
        harness.render_system_prompt("/usr/bin/python3")
    except (KeyError, IndexError, ValueError) as error:
        errors.append(f"the system prompt does not format: {error!r}; it must contain `{{python}}` once and "
                      "escape other braces as {{ and }}")
    if "{python}" not in harness.system_prompt:
        errors.append("the system prompt must contain the `{python}` placeholder")
    if len(harness.system_prompt) > MAX_PROMPT_CHARS:
        errors.append(f"the system prompt has {len(harness.system_prompt)} characters; keep it under {MAX_PROMPT_CHARS}")
    if len(harness.kernel_tools) > MAX_TOOLS:
        errors.append(f"{len(harness.kernel_tools)} kernel tools; keep at most {MAX_TOOLS}")
    if len(harness.skills) > MAX_SKILLS:
        errors.append(f"{len(harness.skills)} skills; keep at most {MAX_SKILLS}")
    unknown = set(harness.agent_settings.get("stop_tools") or []) - set(harness.kernel_tools)
    if unknown:
        errors.append(f"agent.stop_tools names {sorted(unknown)}, which are not kernel tools of this harness")
    for tool in harness.kernel_tools.values():
        errors += [f"tool {tool.get('name')!r}: {problem}" for problem in check_tool(tool)]
    if errors or not smoke:
        return errors
    return smoke_test(harness)


def check_tool(tool: dict) -> List[str]:
    name, code = tool.get("name") or "", tool.get("code") or ""
    problems = []
    if not name.isidentifier() or name in RESERVED_NAMES:
        problems.append(f"the name must be a Python identifier other than {sorted(RESERVED_NAMES)}")
    if not (tool.get("description") or "").strip():
        problems.append("describe what the tool does: the model only sees the description")
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        problems.append('parameters must be a JSON schema: {"type": "object", "properties": {...}}')
    if len(code) > MAX_CODE_CHARS:
        problems.append(f"keep the code under {MAX_CODE_CHARS} characters")
    if not re.search(rf"^def {re.escape(name)}\(", code, re.M):
        problems.append(f"the code must define `def {name}(...)` at the top level")
    try:
        compile(code, f"<tool {name}>", "exec")
    except SyntaxError as error:
        problems.append(f"SyntaxError: {error}")
    return problems


def smoke_test(harness: Harness) -> List[str]:
    """Runs the harness's kernel setup in a fresh kernel and checks each kernel tool is defined."""
    from jupyter_client.manager import start_new_kernel

    km, kc = start_new_kernel(kernel_name="python3", stderr=subprocess.DEVNULL)
    try:
        def run(code: str) -> str:
            return harness._run_cell(code, 120, km, kc)

        output = run(harness.kernel_setup())
        if "Traceback" in output:
            return [f"the kernel setup (helpers + tool code) failed:\n{output[-2000:]}"]
        names = json.dumps(sorted(harness.kernel_tools))
        output = run(f"print([n for n in {names} if not callable(globals().get(n))])")
        if output.strip() != "[]":
            return [f"kernel tools not defined as functions after setup: {output.strip()}"]
        return []
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
