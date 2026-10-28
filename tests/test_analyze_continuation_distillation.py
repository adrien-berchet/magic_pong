"""Reject stale selections/aliases and check paired outcome accounting."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import analyze_continuation_distillation as export  # noqa: E402


def fixture():
    selection = {
        "_sha256": "sealed",
        "arms": {arm: {"best_trained_epoch": 15} for arm in export.run.ARMS},
    }
    training = {"incumbent_epoch": 15}
    index, results = {}, {}
    for arm in export.POLICIES:
        owner = "incumbent" if arm == "control" else arm
        epoch = 0 if arm == "initial" else 15
        index[arm] = {
            "arm": arm,
            "epoch": epoch,
            "alias_of": "incumbent" if arm == "control" else None,
            "artifact": owner + ".json",
            "tensor_sha256": owner,
        }
        results[arm] = {
            "tensor_sha256": owner,
            "selection": {"arm": owner, "epoch": epoch, "selection_sha256": "sealed"},
        }
    return index, selection, training, results


def test_valid_alias_preserves_distinct_policy_view():
    export.validate_index(*fixture())


@pytest.mark.parametrize(
    "problem",
    (
        "epoch",
        "arm",
        "alias_tensor",
        "alias_artifact",
        "alias_chain",
        "result_tensor",
        "selection_hash",
        "selection_owner",
    ),
)
def test_stale_metadata_and_invalid_aliases_fail(problem):
    index, selection, training, results = copy.deepcopy(fixture())
    if problem == "epoch":
        index["continuation"]["epoch"] = 10
    elif problem == "arm":
        index["continuation"]["arm"] = "control"
    elif problem == "alias_tensor":
        index["control"]["tensor_sha256"] = "different"
    elif problem == "alias_artifact":
        index["control"]["artifact"] = "stale.json"
    elif problem == "alias_chain":
        index["incumbent"]["alias_of"] = "initial"
    elif problem == "result_tensor":
        results["continuation"]["tensor_sha256"] = "different"
    elif problem == "selection_hash":
        results["continuation"]["selection"]["selection_sha256"] = "old"
    else:
        results["control"]["selection"]["arm"] = "control"
    with pytest.raises(ValueError):
        export.validate_index(index, selection, training, results)


def test_paired_goal_win_loss_deltas_and_duplicate_rejection():
    common = {"opponent": "striker", "player_side": 2, "bonuses_enabled": True, "seed": 650100}
    reference = {
        "matches": [
            {**common, "agent_won": False, "timeout": False, "goals_for": 1, "goals_against": 3}
        ]
    }
    challenger = {
        "matches": [
            {**common, "agent_won": True, "timeout": False, "goals_for": 3, "goals_against": 0}
        ]
    }
    assert export.paired(reference, challenger)["total_delta"] == {
        "win_delta": 1,
        "loss_delta": -1,
        "goal_margin_delta": 5,
    }
    with pytest.raises(ValueError, match="paired"):
        export.paired(reference, {"matches": challenger["matches"] * 2})


def test_timeouts_keep_partial_goals_without_becoming_wins():
    match = {"agent_won": False, "timeout": True, "goals_for": 2, "goals_against": 0, "steps": 6000}
    counts = export.total([{"matches": [match]}])
    assert (counts["wins"], counts["losses"], counts["timeouts"]) == (0, 0, 1)
    assert counts["goal_margin"] == 2 and counts["frames"] == 6000
