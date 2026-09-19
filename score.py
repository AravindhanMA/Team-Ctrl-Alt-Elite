"""
Soft-score calculator, reproducing the shape of the validator's own JSON
report (§2.7) as closely as the spec allows.

INTERPRETATION NOTE (flag in your write-up): the spec gives two related but
distinct formulas:
  - Score_A/B/C (§2.5): "sum_tier(priority_weight_tier x overrun_days_tier)"
    -- looks like a simple tier-level sum.
  - priority_weighted_score (§2.7): "contract_weight x (1 + activity_priority)
    x overrun_days, summed per overrunning activity" -- an activity-level
    nudge on top of the tier weight.
These can't both be literal at once (the first ignores activity_priority,
the second doesn't) -- we treat §2.7's formula as authoritative (it's the
one the JSON schema actually reports) and treat the §2.5 formula as its
tier-level shorthand. For Scenario A, objective_score == priority_weighted_score
(no other terms in A's objective).

Since activities don't carry their own deadline (only contracts do), we
attribute a contract's overrun_days to whichever of its activities actually
finished in the contract's completion week (i.e. the activity/activities
that determined that completion date), and use THEIR activity_priority for
the nudge. Activities that finished earlier within an overrunning contract
contribute 0 (they weren't the cause of the overrun).
"""
from __future__ import annotations

import pandas as pd

from data_model import Instance, load_instance, week_start_date

CONTRACT_TIER_WEIGHT = {1: 100, 2: 10, 3: 1}
ACTIVITY_NUDGE = {1: 0.3, 2: 0.2, 3: 0.0}


def compute_soft_scores(inst: Instance, access_df: pd.DataFrame, occ_df: pd.DataFrame,
                         results_df: pd.DataFrame, scenario: str):
    proj = inst.project_details.set_index("contract_number")
    act = inst.activity_details.set_index("activity_id")

    # per-activity last scheduled week (finish week)
    finish_week = access_df.groupby("activity_id")["week"].max()

    overrun_days_total = 0
    earliness_days_total = 0
    contracts_overrunning = 0
    priority_overrun = {1: 0, 2: 0, 3: 0}
    priority_weighted_score = 0.0

    results_idx = results_df.set_index("contract_number")

    for contract_number, contract in proj.iterrows():
        tier = int(contract["contract_priority"])
        planned = pd.Timestamp(contract["planned_completion_date"]).date()
        res = results_idx.loc[contract_number]
        sim_date = res["simulated_completion_date"]
        if pd.isna(sim_date):
            continue
        sim_date = pd.Timestamp(sim_date).date()
        raw_diff = (sim_date - planned).days

        if raw_diff > 0:
            overrun_days_total += raw_diff
            contracts_overrunning += 1
            priority_overrun[tier] += raw_diff

            # which activities actually determined this finish?
            contract_activity_ids = act[act["contract_number"] == contract_number].index
            weeks = {aid: finish_week.get(aid) for aid in contract_activity_ids}
            weeks = {aid: w for aid, w in weeks.items() if w is not None}
            if weeks:
                max_week = max(weeks.values())
                determining = [aid for aid, w in weeks.items() if w == max_week]
                for aid in determining:
                    a_priority = int(act.loc[aid, "activity_priority"])
                    nudge = ACTIVITY_NUDGE[a_priority]
                    priority_weighted_score += CONTRACT_TIER_WEIGHT[tier] * (1 + nudge) * raw_diff
        elif raw_diff < 0:
            earliness_days_total += -raw_diff

    # excess access-nights: additional activities beyond nominal
    # LOCATION_SUPPLY, summed across location-weeks. Computed directly from
    # the occupancy output (ground truth), not scheduler bookkeeping.
    excess_access_nights_total = 0
    if scenario in ("B", "C"):
        grp = occ_df.groupby(["location_id", "week"])["activity_id"].nunique()
        for (loc, week), n_activities in grp.items():
            cap = inst.supply.get(loc, 0)
            if n_activities > cap:
                excess_access_nights_total += n_activities - cap

    eclo_nights_total = int((access_df["eclo"] == 1).sum())

    soft_scores = {
        "scenario": scenario,
        "overrun_days_total": overrun_days_total,
        "contracts_overrunning": contracts_overrunning,
        "earliness_days_total": earliness_days_total,
        "excess_access_nights_total": excess_access_nights_total,
        "eclo_nights_total": eclo_nights_total,
        "priority_overrun": {str(k): v for k, v in priority_overrun.items()},
        "priority_weighted_score": round(priority_weighted_score, 1),
    }

    if scenario == "A":
        objective_score = soft_scores["priority_weighted_score"]
    elif scenario == "B":
        objective_score = 7 * excess_access_nights_total + 5 * eclo_nights_total
    else:  # C
        objective_score = (soft_scores["priority_weighted_score"]
                            + 7 * excess_access_nights_total
                            + 5 * eclo_nights_total)

    return {
        "scenario": scenario,
        "feasible": True,  # caller should AND this with validator's hard_violations==[]
        "soft_scores": soft_scores,
        "objective_score": round(objective_score, 1),
        "detail": {
            "nights_scheduled": len(access_df),
            "eclo_nights": eclo_nights_total,
        },
    }


if __name__ == "__main__":
    from scheduler import schedule_scenario_a, build_output_frames
    from validator import validate
    import json

    inst = load_instance()
    state, info = schedule_scenario_a(inst)
    access_df, occ_df, results_df = build_output_frames(inst, state, info, "A")

    violations = validate(inst, access_df, occ_df, "A")
    report = compute_soft_scores(inst, access_df, occ_df, results_df, "A")
    report["feasible"] = len(violations) == 0
    report["hard_violations"] = violations

    print(json.dumps(report, indent=2, default=str))
