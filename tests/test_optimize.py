import json
import textwrap

import pytest

from ipyh import Harness
from ipyh.harness import HARNESS_FILE
from ipyh_optimize import loop
from ipyh_optimize.evidence import profile_file, quirk_kind
from ipyh_optimize.eval import grade
from ipyh_optimize.report import render_report
from ipyh_optimize.select import make_splits
from ipyh_optimize.tasks import load_tasks
from ipyh_optimize.validate import validate_harness

TOOL = {
    "name": "shout",
    "description": "Prints the text in upper case.",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    "code": "def shout(text):\n    print(text.upper())\n",
}

GRADER = '''
import os, sys
gold = os.environ["EXPECTED_ANSWER"].strip().lower()
print("1.0" if sys.stdin.read().strip().lower() == gold else "0.0")
'''


def make_task(root, name, tier="easy", kind="numeric", gold="42"):
    folder = root / "tasks" / name
    (folder / "tests").mkdir(parents=True)
    (folder / "task.toml").write_text(textwrap.dedent(f"""
        [task]
        description = "How many rows does {name} have?"
        [metadata]
        gold_answer = "{gold}"
        reward_mode_initial = "{kind}"
        difficulty_tier = "{tier}"
        [environment.env]
        HF_BUCKET = "someone/bucket"
        BUCKET_PREFIX = "{name}-data"
        [verifier.env]
        EXPECTED_ANSWER = "{gold}"
        [agent]
        timeout_sec = 60.0
    """))
    (folder / "instruction.md").write_text(
        "Files (in /home/user/input, no subfolders):\n- data.csv\n- extra.csv\n\nQuestion:\n...\n"
        "Write only that value to /workdir/answer.txt\n"
    )
    (folder / "tests" / "grader.py").write_text(GRADER)
    return folder


@pytest.fixture
def suite(tmp_path):
    for i in range(12):
        make_task(tmp_path, f"t{i:02d}", tier=("easy", "hard")[i % 2], kind=("numeric", "list", "exact_short")[i % 3])
    return tmp_path


def test_harness_folder_round_trip(tmp_path):
    skill = tmp_path / "src-skills" / "csv"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: csv\ndescription: Reading odd CSV files.\n---\nUse encoding.\n")
    harness = Harness(system_prompt="Be brief. {python}", tools=[TOOL], skills=[skill],
                      cell_timeout_sec=60, agent_settings={"max_iteration": 20})
    folder = harness.save(tmp_path / "h")

    config = json.loads((folder / HARNESS_FILE).read_text())
    assert config["tools"][0]["code_file"] == "tools/shout.py"
    assert (folder / "skills" / "csv" / "SKILL.md").is_file()

    loaded = Harness.from_config(folder)
    assert loaded.system_prompt == "Be brief. {python}"
    assert loaded.kernel_tools["shout"]["code"] == TOOL["code"]
    assert [s.name for s in loaded.skills] == ["csv"]
    assert loaded.cell_timeout_sec == 60
    assert loaded.agent_settings == {"max_iteration": 20}
    assert validate_harness(folder, smoke=False) == []


def test_from_config_rejects_unknown_agent_setting(tmp_path):
    (tmp_path / HARNESS_FILE).write_text(json.dumps({"agent": {"max_turns": 3}}))
    with pytest.raises(ValueError, match="unknown agent settings"):
        Harness.from_config(tmp_path)


def test_validate_reports_problems(tmp_path):
    bad_tool = {**TOOL, "name": "view", "code": "def other(:\n"}
    Harness(system_prompt="No placeholder {here}", tools=[bad_tool]).save(tmp_path / "h")
    errors = validate_harness(tmp_path / "h", smoke=False)
    assert any("{python}" in e for e in errors)
    assert any("does not format" in e for e in errors)
    assert any("identifier other than" in e for e in errors)
    assert any("SyntaxError" in e for e in errors)


def test_validate_smoke_test_runs_tool_code(tmp_path):
    Harness(tools=[TOOL]).save(tmp_path / "ok")
    assert validate_harness(tmp_path / "ok") == []
    Harness(tools=[{**TOOL, "code": "import not_a_module\ndef shout(text):\n    pass\n"}]).save(tmp_path / "bad")
    assert "kernel setup" in validate_harness(tmp_path / "bad")[0]


def test_task_card_is_label_free(suite):
    task = load_tasks(suite)[0]
    card = task.card()
    assert card["files"] == ["data.csv", "extra.csv"]
    assert card["answer_kind"] == "numeric"
    assert "42" not in json.dumps(card)
    assert not {"gold_answer", "EXPECTED_ANSWER", "verifier"} & set(card)


def test_splits_are_disjoint_and_stratified(suite):
    splits = make_splits(load_tasks(suite), n_train=6, n_heldout=4, rounds=2)
    assert not set(splits["train"]) & set(splits["heldout"])
    assert sorted(sum(splits["rounds"], [])) == sorted(splits["train"])
    tiers = [t.metadata["difficulty_tier"] for t in load_tasks(suite, splits["heldout"])]
    assert set(tiers) == {"easy", "hard"}


def test_grade_uses_the_task_grader(suite, tmp_path):
    task = load_tasks(suite)[0]
    assert grade(task, "42", tmp_path / "g.log") == 1.0
    assert grade(task, "41", tmp_path / "g.log") == 0.0


def test_profile_file_finds_quirks(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes("name;price;when\n Café ;\"1,200\";2024-01-02\nB;?;2024-01-03\n".encode("latin-1"))
    profile = profile_file(path)
    kinds = {quirk_kind(q) for q in profile["quirks"]}
    assert profile["delimiter"] == ";"
    assert {"not UTF-8", "delimiter is not a comma", "dates stored as text"} <= kinds


def test_manual_step_waits_then_validates(suite, tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "build_evidence", lambda tasks, base, out, cache: out.mkdir(parents=True))
    run = loop.init_run(tmp_path / "run", suite, n_train=4, n_heldout=2, cache=tmp_path / "cache")
    assert loop.step(run, editor=None) is False  # waits for the INV edit
    prompt = (run / "round-1" / "prompts" / "inv.md").read_text()
    assert "INVARIANT" in prompt and str(run / "round-1" / "inv") in prompt and "{{" not in prompt.split("`{{`")[0]

    # The "editor" breaks the candidate: the next step reports it and keeps waiting
    (run / "round-1" / "inv" / "system_prompt.md").write_text("no placeholder")
    assert loop.step(run, editor=None) is False
    (run / "round-1" / "inv" / "system_prompt.md").write_text("Fixed. {python}")
    for _ in range(6):  # inv validated, then dom, compose and reflect each wait once
        if loop.step(run, editor=None):
            break
        library = run / "library" / "EXPERIENCE.md"
        if library.exists():
            library.write_text("## csv\nCue: x\nLesson: y\nEvidence: z\n")
    assert (run / "harness" / HARNESS_FILE).is_file()
    assert json.loads((run / "harness" / HARNESS_FILE).read_text())["meta"]["optimized_by"] == "ipyh-optimize"
    assert "Fixed." not in (run / "harness" / "system_prompt.md").read_text()  # composer kept the base
    assert (run / "round-1" / "diffs" / "inv.diff").read_text().count("Fixed.") == 1


def test_editor_command_fallback(suite, tmp_path, monkeypatch):
    """A scripted editor that writes an invalid prompt: one retry, then the base is kept."""
    monkeypatch.setattr(loop, "build_evidence", lambda tasks, base, out, cache: out.mkdir(parents=True))
    run = loop.init_run(tmp_path / "run", suite, n_train=4, n_heldout=2, cache=tmp_path / "cache")
    editor = ("sh -c 'test -f harness.json && echo broken > system_prompt.md; "
              "test -f EXPERIENCE.md && printf \"Lesson: x\\n\" >> EXPERIENCE.md; true'")
    assert loop.step(run, editor=editor) is True
    assert "fell back" in (run / "round-1" / "inv" / "CHANGELOG.md").read_text()
    assert validate_harness(run / "harness", smoke=False) == []


def test_report_pairs_tasks():
    rows = lambda name, rewards, turns: [
        {"harness": name, "task": f"t{i}", "repeat": 0, "reward": r, "turns": t, "answered": True,
         "timed_out": False, "tool_errors": 0, "prompt_tokens": 10, "completion_tokens": 5,
         "difficulty": "easy", "answer_kind": "numeric"}
        for i, (r, t) in enumerate(zip(rewards, turns))]
    report = render_report({"base": rows("base", [0, 1, 0, 1], [8, 6, 9, 7]),
                            "optimized": rows("optimized", [1, 1, 1, 1], [3, 3, 4, 3])})
    assert "| base | 50.0%" in report and "| optimized | 100.0%" in report
    assert "pass rate: +50.0%" in report and "solved only by optimized: 2" in report


def test_stop_tool_ends_the_run_only_on_success(tmp_path):
    import asyncio
    from types import SimpleNamespace

    submit = {"name": "submit", "description": "Submit.", "code": (
        "def submit(value):\n    if not value:\n        raise ValueError('empty')\n    print('ok')\n"),
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}

    class Model:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=self)

        def with_options(self, **_):
            return self

        async def create(self, model, messages, tools, max_tokens):
            self.calls += 1
            call = {"id": f"c{self.calls}", "type": "function",
                    "function": {"name": "submit", "arguments": json.dumps({"value": "" if self.calls == 1 else "42"})}}
            message = SimpleNamespace(content="", tool_calls=[SimpleNamespace(model_dump=lambda: call)], model_extra={})
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")], usage=usage)

    folder = Harness(tools=[submit], agent_settings={"stop_tools": ["submit"], "max_iteration": 5}).save(tmp_path / "h")
    assert validate_harness(folder, smoke=False) == []
    model = Model()
    from ipyh import Agent

    harness = Harness.from_config(folder)
    agent = Agent(model, harness, model_name="m", **harness.agent_settings)
    asyncio.run(agent.run_task("submit 42"))
    # The failed (raising) call does not stop the run; the successful one does, with no closing turn
    assert model.calls == 2
    assert agent.trajectories[agent.current_task]["done"]

    config = json.loads((folder / HARNESS_FILE).read_text())
    config["agent"]["stop_tools"] = ["missing"]
    (folder / HARNESS_FILE).write_text(json.dumps(config))
    assert any("not kernel tools" in e for e in validate_harness(folder, smoke=False))
