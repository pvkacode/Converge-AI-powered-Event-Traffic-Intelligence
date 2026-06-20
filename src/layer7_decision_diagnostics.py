"""
Layer 7 — M7C.1: Decision-Quality Diagnostics (transparency only).

Read-only audit over the M7C control outputs (action_scores + recommendations). Computes
the Action Competition Audit (Part A) and Counterfactual Regret Diagnostics (Part B).
It NEVER alters a recommendation, an action score, the optimizer, spillover, centrality,
constraints, priority, or approval logic — it only reads and explains them.

NOTE (naming): Part B7's requested 'layer7_decision_confidence.csv' collides with the
existing M5 Decision Confidence Score output, so the M7C.1 decision-confidence diagnostic
is written to 'layer7_control_decision_confidence.csv' to preserve the M5 file unchanged.

ADDITIVE ONLY. New outputs only:
  layer7_action_competition_audit.csv     layer7_action_competition_matrix.csv
  layer7_action_efficiency.csv            layer7_counterfactual_actions.csv
  layer7_control_decision_confidence.csv  layer7_override_regret.csv
  layer7_decision_quality_summary.txt
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_EPS = 1e-9
_DOMINANCE = 0.70

CATEGORY = {
    "SIGNAL_EXTEND_GREEN": "signal", "SIGNAL_REDUCE_GREEN": "signal", "SIGNAL_OFFSET_ADJUST": "signal",
    "ACTIVATE_DIVERSION": "diversion", "DEACTIVATE_DIVERSION": "diversion",
    "QUEUE_RELIEF": "queue_relief",
    "DISPATCH_POLICE": "dispatch", "DISPATCH_MARSHAL": "dispatch",
    "DISPATCH_TOW": "dispatch", "DISPATCH_QRU": "dispatch",
    "VMS_MESSAGE": "information", "OPERATOR_ESCALATION": "information", "ROADWORK_ALERT": "information",
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


# --------------------------------------------------------------- Part A
def action_competition(scores: pd.DataFrame, recs: pd.DataFrame):
    selected = recs["recommended_action"].value_counts().to_dict()
    total_sel = int(recs["recommended_action"].ne("NO_ACTION").sum())
    by_site_cat_best = {}  # (event_id, category) -> best control_score in that category at site
    for (eid, cat), g in scores.assign(cat=scores["action_id"].map(CATEGORY)).groupby(["event_id", "cat"]):
        by_site_cat_best[(eid, cat)] = float(g["control_score"].max())

    rows = []
    for aid, g in scores.groupby("action_id"):
        cand = int(len(g)); sel = int(selected.get(aid, 0))
        wins = {"signal": 0, "diversion": 0, "queue_relief": 0, "dispatch": 0}
        for _, r in g.iterrows():
            eid = r["event_id"]; cs = float(r["control_score"])
            for cat in wins:
                if CATEGORY[aid] == cat:
                    continue
                best = by_site_cat_best.get((eid, cat))
                if best is not None and cs > best:
                    wins[cat] += 1
        rows.append({
            "action_id": aid, "category": CATEGORY[aid],
            "candidate_count": cand, "selected_count": sel,
            "selection_rate": round(sel / cand, 4) if cand else 0.0,
            "share_selected": round(sel / total_sel, 4) if total_sel else 0.0,
            "mean_benefit": round(float(g["benefit_score"].mean()), 6),
            "median_benefit": round(float(g["benefit_score"].median()), 6),
            "mean_cost": round(float(g["cost_score"].mean()), 6),
            "median_cost": round(float(g["cost_score"].median()), 6),
            "mean_control_score": round(float(g["control_score"].mean()), 6),
            "median_control_score": round(float(g["control_score"].median()), 6),
            "wins_against_signal": wins["signal"], "wins_against_diversion": wins["diversion"],
            "wins_against_queue_relief": wins["queue_relief"], "wins_against_dispatch": wins["dispatch"],
            "dominance_flag": "POTENTIAL_OPTIMIZER_DOMINANCE" if (total_sel and sel / total_sel > _DOMINANCE)
                              else "NO_DOMINANCE_DETECTED",
            "generated_at": _NOW_ISO,
        })
    audit = pd.DataFrame(rows).sort_values("selected_count", ascending=False)
    return audit, total_sel


def competition_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    by_site = {eid: dict(zip(g["action_id"], g["control_score"]))
               for eid, g in scores.groupby("event_id")}
    actions = sorted(scores["action_id"].unique())
    rows = []
    for a in actions:
        for b in actions:
            if a == b:
                continue
            common = a_wins = b_wins = ties = 0
            for eid, m in by_site.items():
                if a in m and b in m:
                    common += 1
                    if m[a] > m[b]:
                        a_wins += 1
                    elif m[b] > m[a]:
                        b_wins += 1
                    else:
                        ties += 1
            if common:
                rows.append({"action_a": a, "action_b": b, "n_common_sites": common,
                             "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
                             "a_win_rate": round(a_wins / common, 4), "generated_at": _NOW_ISO})
    return pd.DataFrame(rows)


def action_efficiency(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    s["efficiency"] = s["benefit_score"] / s["cost_score"].clip(lower=_EPS)
    rows = []
    for aid, g in s.groupby("action_id"):
        rows.append({"action_id": aid, "category": CATEGORY[aid], "n": int(len(g)),
                     "mean_efficiency": round(float(g["efficiency"].mean()), 6),
                     "median_efficiency": round(float(g["efficiency"].median()), 6),
                     "mean_benefit": round(float(g["benefit_score"].mean()), 6),
                     "mean_cost": round(float(g["cost_score"].mean()), 6),
                     "generated_at": _NOW_ISO})
    return pd.DataFrame(rows).sort_values("mean_efficiency", ascending=False)


# --------------------------------------------------------------- Part B
def _tier(c: float) -> str:
    if c >= 0.50:
        return "VERY_STRONG"
    if c >= 0.30:
        return "STRONG"
    if c >= 0.15:
        return "MODERATE"
    return "WEAK"


def counterfactual(scores: pd.DataFrame):
    cf_rows, dc_rows, regret_rows = [], [], []
    for eid, g in scores.groupby("event_id"):
        gg = g.sort_values("control_score", ascending=False).reset_index(drop=True)
        best_a, best_s = gg.loc[0, "action_id"], float(gg.loc[0, "control_score"])
        if len(gg) >= 2:
            sec_a, sec_s = gg.loc[1, "action_id"], float(gg.loc[1, "control_score"])
        else:
            sec_a, sec_s = "NO_ACTION", 0.0
        gap = best_s - sec_s
        rel_gap = gap / max(abs(best_s), _EPS)
        conf = float(np.clip(gap / max(abs(best_s), _EPS), 0.0, 1.0))
        review = conf < 0.15

        # regrets vs every alternative (best - alt)
        regrets = []
        for _, r in gg.iloc[1:].iterrows():
            reg = best_s - float(r["control_score"])
            regrets.append(reg)
            regret_rows.append({"event_id": eid, "best_action": best_a, "best_score": round(best_s, 6),
                                "override_action": r["action_id"], "override_score": round(float(r["control_score"]), 6),
                                "regret_if_overridden": round(reg, 6), "generated_at": _NOW_ISO})
        rmin = round(float(min(regrets)), 6) if regrets else 0.0
        rmax = round(float(max(regrets)), 6) if regrets else 0.0
        rmean = round(float(np.mean(regrets)), 6) if regrets else 0.0

        cf_rows.append({"event_id": eid, "best_action": best_a, "best_score": round(best_s, 6),
                        "second_best_action": sec_a, "second_best_score": round(sec_s, 6),
                        "score_gap_abs": round(gap, 6), "score_gap_rel": round(rel_gap, 6),
                        "min_regret": rmin, "max_regret": rmax, "mean_regret": rmean,
                        "n_candidates": int(len(gg)), "generated_at": _NOW_ISO})
        dc_rows.append({"event_id": eid, "best_action": best_a,
                        "best_score": round(best_s, 6), "second_best_score": round(sec_s, 6),
                        "score_gap_abs": round(gap, 6), "score_gap_rel": round(rel_gap, 6),
                        "decision_confidence": round(conf, 6), "decision_stability_tier": _tier(conf),
                        "operator_attention_flag": "REVIEW_RECOMMENDED" if review else "OK",
                        "generated_at": _NOW_ISO})
    return pd.DataFrame(cf_rows), pd.DataFrame(dc_rows), pd.DataFrame(regret_rows)


# --------------------------------------------------------------- run
def run(write: bool = True) -> tuple[dict, list[dict]]:
    scores = _read("layer7_control_action_scores.csv")
    recs = _read("layer7_control_recommendations.csv")

    audit, total_sel = action_competition(scores, recs)
    matrix = competition_matrix(scores)
    eff = action_efficiency(scores)
    cf, dc, regret = counterfactual(scores)

    if write:
        audit.to_csv(OUT / "layer7_action_competition_audit.csv", index=False)
        matrix.to_csv(OUT / "layer7_action_competition_matrix.csv", index=False)
        eff.to_csv(OUT / "layer7_action_efficiency.csv", index=False)
        cf.to_csv(OUT / "layer7_counterfactual_actions.csv", index=False)
        dc.to_csv(OUT / "layer7_control_decision_confidence.csv", index=False)
        regret.to_csv(OUT / "layer7_override_regret.csv", index=False)

    checks = _validate(scores, recs, audit, matrix, cf, dc, regret)
    if write:
        _summary(audit, dc, cf, total_sel, checks)
    return {"audit": audit, "matrix": matrix, "efficiency": eff,
            "counterfactual": cf, "decision_confidence": dc, "regret": regret}, checks


def _validate(scores, recs, audit, matrix, cf, dc, regret) -> list[dict]:
    checks = []

    def chk(cid, passed, detail, sev="critical"):
        checks.append({"check_id": cid, "phase": "decision_diagnostics", "passed": bool(passed),
                       "detail": detail, "severity": "info" if passed else sev})

    # best_action (from scores) matches the existing recommendation -> recommendations unchanged
    rec_map = dict(zip(recs["event_id"].astype(str), recs["recommended_action"]))
    cf_map = dict(zip(cf["event_id"].astype(str), cf["best_action"]))
    mism = [e for e in cf_map if rec_map.get(e) not in (cf_map[e],) and rec_map.get(e) != "NO_ACTION"]
    chk("m7c1_best_matches_recommendation", len(mism) == 0,
        f"{len(mism)} sites where diagnostic best != existing recommendation (should be 0)")
    chk("m7c1_confidence_bounded", bool(((dc["decision_confidence"] >= 0)
        & (dc["decision_confidence"] <= 1)).all()),
        f"decision_confidence in [{dc['decision_confidence'].min():.3f},{dc['decision_confidence'].max():.3f}]")
    chk("m7c1_confidence_finite", bool(np.isfinite(dc["decision_confidence"]).all()), "confidence finite")
    chk("m7c1_regret_finite_nonneg", bool(np.isfinite(regret["regret_if_overridden"]).all()
        and (regret["regret_if_overridden"] >= -1e-9).all()),
        f"regret finite, non-negative; range [{regret['regret_if_overridden'].min():.4f},"
        f"{regret['regret_if_overridden'].max():.4f}]")
    valid_pairs = bool(((matrix["a_wins"] + matrix["b_wins"] + matrix["ties"]) == matrix["n_common_sites"]).all())
    chk("m7c1_pairwise_counts_valid", valid_pairs, "a_wins+b_wins+ties == n_common for every pair")
    return checks


def _summary(audit, dc, cf, total_sel, checks) -> None:
    n_pass = sum(1 for c in checks if c["passed"]); n_fail = sum(1 for c in checks if not c["passed"])
    dom = audit[audit["share_selected"] > _DOMINANCE]
    weak = int((dc["decision_stability_tier"] == "WEAK").sum())
    review = int((dc["operator_attention_flag"] == "REVIEW_RECOMMENDED").sum())
    top_regret = cf.sort_values("max_regret", ascending=False).head(3)
    low_regret = cf.sort_values("max_regret").head(3)
    sel = audit[audit["selected_count"] > 0][["action_id", "selected_count", "share_selected",
                                              "mean_control_score", "mean_cost"]]
    lines = [
        "LAYER 7 — M7C.1 DECISION-QUALITY SUMMARY",
        "=" * 46,
        f"generated_at: {_NOW_ISO}",
        "transparency only — no recommendation, score, or optimizer logic was changed.",
        "",
        "ACTION DISTRIBUTION (selected):",
        "  " + str(dict(zip(sel["action_id"], sel["selected_count"]))),
        "",
        "COMPETITION (selected actions):",
    ]
    for _, r in sel.iterrows():
        a = audit[audit.action_id == r["action_id"]].iloc[0]
        lines.append(f"  {r['action_id']:22s} share={r['share_selected']:.2f} "
                     f"mean_control={r['mean_control_score']:.3f} "
                     f"wins(sig/div/qr/disp)={a['wins_against_signal']}/{a['wins_against_diversion']}/"
                     f"{a['wins_against_queue_relief']}/{a['wins_against_dispatch']}")
    lines += [
        "",
        "DOMINANCE: " + ("POTENTIAL_OPTIMIZER_DOMINANCE — " + ", ".join(dom["action_id"])
                         if len(dom) else "NO_DOMINANCE_DETECTED "
                         f"(max share = {audit['share_selected'].max():.2f} < {_DOMINANCE})"),
        "",
        f"mean decision confidence: {dc['decision_confidence'].mean():.4f}",
        f"decision stability tiers: {dc['decision_stability_tier'].value_counts().to_dict()}",
        f"weak-decision rate: {weak}/{len(dc)} = {weak/len(dc):.3f}",
        f"review-recommended count: {review}",
        "",
        "HIGHEST-REGRET decisions (max_regret):",
    ]
    for _, r in top_regret.iterrows():
        lines.append(f"  {r['event_id']}: best={r['best_action']} max_regret={r['max_regret']:.3f}")
    lines += ["", "LOWEST-REGRET decisions:"]
    for _, r in low_regret.iterrows():
        lines.append(f"  {r['event_id']}: best={r['best_action']} max_regret={r['max_regret']:.3f}")
    lines += ["", f"VALIDATION: {n_pass} passed / {n_fail} failed"]
    for c in checks:
        lines.append(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['check_id']}: {c['detail']}")
    (OUT / "layer7_decision_quality_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 56)
    print("LAYER 7 — M7C.1 DECISION-QUALITY DIAGNOSTICS")
    print("=" * 56)
    tables, checks = run(write=True)
    dc = tables["decision_confidence"]
    print(f"mean decision_confidence: {dc['decision_confidence'].mean():.4f}; "
          f"tiers={dc['decision_stability_tier'].value_counts().to_dict()}")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
    n_fail = sum(1 for c in checks if not c["passed"])
    print(f"\nValidation: {len(checks)-n_fail} passed / {n_fail} failed")


if __name__ == "__main__":
    main()
