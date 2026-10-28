"""Protocol guards plus archived real-engine replay/counterfactual parity."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_striker_scoring_diagnostic as diagnostic  # noqa: E402


def test_fixed_panel_has_all_eight_unique_cells():
    cells = diagnostic.cells()
    assert len(cells) == len({diagnostic.key(c) for c in cells}) == 8
    assert {c["side"] for c in cells} == {1, 2}
    assert {c["bonuses"] for c in cells} == {False, True}
    assert {c["match_seed"] for c in cells} == {620000, 620001}


def test_replay_rejects_even_single_field_drift():
    recorded = {"winner": 0, "steps": 6000, "goals_for": 0, "goals_against": 2}
    diagnostic.assert_replay(recorded.copy(), recorded)
    with pytest.raises(AssertionError, match="did not reproduce"):
        diagnostic.assert_replay({**recorded, "goals_for": 1}, recorded)


def test_regret_anchor_uses_earliest_tie():
    assert diagnostic.replace_worst(None, {"regret": 0})
    assert diagnostic.replace_worst({"regret": 1}, {"regret": 2})
    assert not diagnostic.replace_worst({"regret": 2}, {"regret": 2})
    assert not diagnostic.replace_worst({"regret": 2}, {"regret": 1})


def test_event_accounting_is_relative_to_side():
    events = {"goals": [{"player": 2}], "paddle_hits": [{"player": 1}, {"player": 2}]}
    assert diagnostic.event_summary(events, 2) == {
        "goals_for": 1,
        "goals_against": 0,
        "own_hits": 1,
        "other_hits": 1,
    }


def test_changed_pinned_input_is_rejected(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("pinned")
    manifest = {
        "schema": diagnostic.SCHEMA,
        "settings": diagnostic.SETTINGS,
        "input_sha256": {str(source): diagnostic.sha(source)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    diagnostic.read_manifest(tmp_path)
    source.write_text("changed")
    # codespell:ignore-next-line
    with pytest.raises(ValueError, match="[Hh]ash|[Cc]hanged|[Ii]ntegrity"):
        diagnostic.read_manifest(tmp_path)


def test_complete_requires_each_named_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "read_manifest", lambda path: {})
    for seed in (11, 22, 44):
        p = tmp_path / f"seed_{seed}/completion.json"
        p.parent.mkdir()
        p.write_text("{}")
    (tmp_path / "teacher_completion.json").write_text("{}")
    with pytest.raises(ValueError, match="Missing completed"):
        diagnostic.verify(tmp_path, complete=True)


def test_archived_player2_bonus_replay_and_branch_parity():
    # Isolated process prevents live/archive module mixing by the test runner.
    script = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path("docs/ai").resolve()))
import run_striker_scoring_diagnostic as d
old = d.combo.read_manifest(d.SOURCE)
d.combo.prior.activate_archive(old)
from magic_pong.ai.competitive import _standard_configuration
from magic_pong.ai.models.planner_ai import make_teacher
from magic_pong.utils.config import game_config
original = d.combo.prior.load_agent(Path(old["models"]["11"]))
geometry = d.combo.previous.ContactGeometryConfig(**old["geometry"])
selection = json.loads((d.SOURCE / "seed_11/selection.json").read_text())
agent, _ = d.policy(original, geometry, selection, 11, "geometry_cost")
cell = {"side": 2, "bonuses": True, "match_seed": 620000}
teacher = make_teacher(90, 30, backend="fast")
actual, trace, details, anchors = d.trace_match(agent, cell, teacher, True)
rows = json.loads((d.SOURCE / "seed_11/heldout_geometry_cost.json").read_text())["matches"]
previous = next(r for r in rows if r["opponent"] == "striker" and r["player_side"] == 2
                and r["bonuses_enabled"] and r["seed"] == 620000)
d.assert_replay(actual, previous)
assert anchors and details["labels"]
with _standard_configuration(d.config(cell)):
    game_config.BONUSES_ENABLED = True
    before = d.rng_state()
    baseline = d.branch(anchors[0], agent, teacher, "student", trace)
    after = d.rng_state()
    assert before[0] == after[0]
    assert (before[1][1] == after[1][1]).all()
    assert (before[2] == after[2]).all()
    events = details["events"][anchors[0]["frame"]:anchors[0]["frame"] + baseline["frames"]]
    for field in ("goals_for", "goals_against", "own_hits", "other_hits"):
        assert baseline[field] == sum(e[field] for e in events)
print(json.dumps({"verification_match_frames": actual["steps"],
                  "verification_branch_frames": baseline["frames"], "replay_exact": True}))
"""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
