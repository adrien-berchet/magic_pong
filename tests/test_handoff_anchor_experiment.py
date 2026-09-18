"""Guard reference identity, split isolation, and the new benchmark contrast."""

from copy import deepcopy

import pytest
import torch

from docs.ai.collect_handoff_anchor_bank import validate_bank
from docs.ai.run_handoff_anchor_experiment import paired_comparison


def synthetic_bank():
    cells = [(s, b, t) for s in (1, 2) for b in (False, True) for t in (1, 3, 5, 11)]
    strata = torch.arange(16).repeat_interleave(128)
    states = torch.zeros(2048, 33)
    states[:, 0] = 0.3
    states[:, 4] = 0.2
    states[:, 2] = -0.5
    states[:, 32] = torch.tensor([cells[i][2] / 11 for i in strata])
    actions = torch.arange(2048) % 9
    metadata = {
        "reference_training_seed": 11,
        "reference_frames": 500000,
        "reference_sha256": "a" * 64,
        "reference_network_sha256": "b" * 64,
        "used_collection_seeds": [83041],
        "rows_provenance": [
            {
                "cell": int(c),
                "seed": 83041,
                "side": cells[c][0],
                "bonuses": cells[c][1],
                "target": cells[c][2],
                "frame": 100,
                "action": int(a),
                "contact_delay": 10,
            }
            for c, a in zip(strata, actions)
        ],
    }
    return {
        "schema": "magic_pong.policy_anchor_bank.v1",
        "states": states,
        "actions": actions,
        "strata": strata,
        "metadata": metadata,
    }


def test_bank_rejects_wrong_teacher_or_validation_seed():
    bank = synthetic_bank()
    validate_bank(bank, 11, "a" * 64, "b" * 64)
    for seed, sha, network in (
        (22, "a" * 64, "b" * 64),
        (11, "c" * 64, "b" * 64),
        (11, "a" * 64, "c" * 64),
    ):
        with pytest.raises(ValueError, match="reference"):
            validate_bank(bank, seed, sha, network)
    for location in ("used", "row"):
        changed = deepcopy(bank)
        if location == "used":
            changed["metadata"]["used_collection_seeds"] = [101]
        else:
            changed["metadata"]["rows_provenance"][0]["seed"] = 101
        with pytest.raises(ValueError, match="seed"):
            validate_bank(changed, 11, "a" * 64, "b" * 64)


@pytest.mark.parametrize("corruption", ["target", "action", "outgoing", "old_contact", "imbalance"])
def test_bank_rejects_inconsistent_defensive_rows(corruption):
    bank = synthetic_bank()
    if corruption == "target":
        bank["states"][0, 32] = 1
    elif corruption == "action":
        bank["metadata"]["rows_provenance"][0]["action"] = 8
    elif corruption == "outgoing":
        bank["states"][0, 2] = 1
    elif corruption == "old_contact":
        bank["metadata"]["rows_provenance"][0]["contact_delay"] = 30
    else:
        bank["strata"][0] = 1
    with pytest.raises(ValueError):
        validate_bank(bank, 11, "a" * 64, "b" * 64)


def test_paired_comparisons_keep_control_and_original_anchor_separate():
    runs = []
    for seed in (11, 22, 33):
        for variant, wins, margin in (
            ("control", 40, -1),
            ("original_anchor", 60, 2),
            ("handoff_anchor", 50, 1),
        ):
            runs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "final": {"wins": wins},
                    "evaluations": [
                        {
                            "matches": [
                                {
                                    "opponent": "follow_ball",
                                    "goals_for": 5 + margin,
                                    "goals_against": 5,
                                }
                            ]
                        }
                    ],
                }
            )
    vs_control = paired_comparison(runs, "control")
    vs_original = paired_comparison(runs, "original_anchor")
    assert vs_control["primary_rule_passed"] and vs_control["secondary_rule_passed"]
    assert vs_control["pooled_win_difference"] == 30
    assert not vs_original["primary_rule_passed"] and not vs_original["secondary_rule_passed"]
    assert vs_original["pooled_win_difference"] == -30
