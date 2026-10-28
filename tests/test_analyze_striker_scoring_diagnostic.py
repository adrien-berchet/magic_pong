"""Check goal/censor accounting and paired branch comparisons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import analyze_striker_scoring_diagnostic as export  # noqa: E402


def test_timeout_partial_goals_are_not_wins_and_contacts_are_side_relative():
    hit = {
        "player": 2,
        "goal_in_frame": False,
        "offset": -0.75,
        "abs_vertical_speed": 600.0,
        "speed": 800.0,
    }
    row = {
        "match": {
            "agent_won": False,
            "timeout": True,
            "goals_for": 1,
            "goals_against": 2,
            "steps": 6000,
        },
        "cell": {"side": 2},
        "labels": [
            {
                "student_action": 0,
                "teacher_action": 1,
                "regret": 750.0,
                "teacher_predicts_goal": False,
                "any_predicted_goal": True,
            }
        ],
        "contacts": [hit, {**hit, "player": 1}, {**hit, "goal_in_frame": True}],
    }
    result = export.aggregate([row])
    assert (result["wins"], result["losses"], result["timeouts"]) == (0, 0, 1)
    assert (result["goals_for"], result["goals_against"]) == (1, 2)
    assert result["own_contacts"] == result["vertical_speed_above_500"] == 1
    assert result["absolute_contact_offset"]["mean"] == 0.75
    assert result["costly_errors_500"] == result["any_predicted_goals"] == 1
    assert result["teacher_predicted_goals"] == result["teacher_action_agreements"] == 0


def test_branch_summary_counts_paired_rescues_and_harms():
    def anchor(outcomes):
        return {
            "branches": [
                {
                    "mode": mode,
                    "goals_for": goal == 1,
                    "goals_against": goal == -1,
                    "censored": goal == 0,
                    "frames": 100,
                }
                for mode, goal in zip(export.run.SETTINGS["branch_modes"], outcomes)
            ]
        }

    summary = export.summarize_branches([anchor([-1, 0, 1, 1]), anchor([1, -1, 1, 0])])
    assert summary["teacher"]["new_goals_vs_student"] == 1
    assert summary["teacher"]["lost_goals_vs_student"] == 1
    assert summary["teacher"]["avoided_concessions"] == 1
    assert summary["teacher_once"]["new_concessions"] == 1
    assert summary["teacher_commit30"]["goals_for"] == 2
    assert summary["student"]["frames"] == 200


def test_empty_distribution_does_not_invent_zero_mean():
    assert export.distribution([]) == {"n": 0}
