"""
Layer 7 — M7C: MPC-lite Control Recommendation Engine (orchestrator).

Converts M7B estimated state / forecasts / escalation / topology + Layer 5 resources +
Layer 6 health into human-in-the-loop traffic-control recommendations. Receding-horizon
objective, topology spillover, surrogate action simulation, resource feasibility, operator
priority, betweenness centrality, and metric-grounded explanations.

Human-in-the-loop: every non-information recommendation requires approval; NOTHING is
executed. Additive, new files only. NO Layer 1-6 / existing Layer 7 output modified.
NO GNN, NO real-time API, NO ML/RL.

Outputs (9 + diagnostics):
  layer7_control_recommendations.csv  layer7_control_action_scores.csv
  layer7_control_simulation.csv       layer7_control_constraints.csv
  layer7_operator_actions.csv         layer7_operator_priority.csv
  layer7_control_explanations.csv     layer7_topology_centrality.csv
  layer7_control_summary.txt          layer7_control_diagnostics.csv
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import layer7_control_explanations as expl
from layer7_config import OUT
from layer7_control_actions import action_dicts, INFORMATION_ONLY, is_applicable
from layer7_control_constraints import apply_resource_feasibility
from layer7_control_optimizer import baseline_J, compute_spillover, score_action
from layer7_control_simulator import simulate_action, whatif

_NOW_ISO = datetime.now(timezone.utc).isoformat()
_EDGE_MIN = 0.05  # significant-edge threshold (matches topology metrics)


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _norm(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 1e-12 else pd.Series(0.0, index=s.index)


# --------------------------------------------------------------- Step 11: centrality
def _betweenness(nodes, edges_sig) -> dict:
    """Pure Brandes betweenness on the unweighted significant-edge graph (50 nodes)."""
    adj = defaultdict(list)
    for a, b in edges_sig:
        adj[a].append(b); adj[b].append(a)
    bet = {n: 0.0 for n in nodes}
    for s in nodes:
        S, P, sigma, dist = [], defaultdict(list), defaultdict(float), {}
        sigma[s] = 1.0; dist[s] = 0
        Qd = deque([s])
        while Qd:
            v = Qd.popleft(); S.append(v)
            for w in adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1; Qd.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = defaultdict(float)
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                bet[w] += delta[w]
    n = len(nodes)
    scale = ((n - 1) * (n - 2)) if n > 2 else 1
    return {k: (v / scale) for k, v in bet.items()}  # directed-pair normalization (undirected counts each twice)


def build_centrality(topo: pd.DataFrame) -> pd.DataFrame:
    nodes = sorted(set(topo["site_a"].astype(str)) | set(topo["site_b"].astype(str)))
    deg = {n: 0 for n in nodes}; wdeg = {n: 0.0 for n in nodes}
    edges_sig = []
    corridor = {}
    for _, e in topo.iterrows():
        a, b, w = str(e["site_a"]), str(e["site_b"]), float(e["adjacency_weight"])
        wdeg[a] += w; wdeg[b] += w
        if w >= _EDGE_MIN:
            deg[a] += 1; deg[b] += 1; edges_sig.append((a, b))
    bet = _betweenness(nodes, edges_sig)
    n = max(1, len(nodes) - 1)
    # corridor rank: rank within corridor by weighted_degree (needs corridor per node)
    cent = _read("layer7_topology_metrics.csv")
    cmap = dict(zip(cent["event_id"].astype(str), cent["corridor"])) if len(cent) else {}
    df = pd.DataFrame([{
        "node_id": i, "site_id": nid, "corridor": cmap.get(nid, "Non-corridor"),
        "degree_centrality": round(deg[nid] / n, 6),
        "weighted_degree_centrality": round(wdeg[nid] / n, 6),
        "betweenness_centrality": round(bet[nid], 6),
        "generated_at": _NOW_ISO,
    } for i, nid in enumerate(nodes)])
    df["corridor_rank"] = (df.groupby("corridor")["weighted_degree_centrality"]
                           .rank(ascending=False, method="min").astype(int))
    return df


# --------------------------------------------------------------- main
def run(write: bool = True) -> tuple[dict, list[dict]]:
    # Step 2: control state from M7B + L5 + L6
    est = _read("layer7_state_estimates.csv")
    wide = est.pivot_table(index="event_id", columns="state_name", values="state_value",
                           aggfunc="first").reset_index()
    wide["event_id"] = wide["event_id"].astype(str)
    cap = _read("layer7_capacity_utilization.csv"); cap["event_id"] = cap["event_id"].astype(str)
    esc = _read("layer7_escalation_risk.csv"); esc["event_id"] = esc["event_id"].astype(str)
    unc = _read("layer7_state_uncertainty.csv")
    fc = _read("layer7_state_forecasts.csv")
    topo = _read("layer7_sensor_topology.csv")
    alloc = _read("layer5_resource_allocation.csv"); alloc["event_id"] = alloc["event_id"].astype(str)

    sites = wide[["event_id", "queue_length", "travel_time", "incident_intensity"]].copy()
    sites = sites.merge(cap[["event_id", "capacity_utilization"]], on="event_id", how="left")
    sites = sites.merge(esc[["event_id", "escalation_risk"]], on="event_id", how="left")
    div = dict(zip(alloc["event_id"],
                   alloc["diversion_activated"].astype(str).str.lower().isin(["1", "true", "yes"])))
    sites["diversion_active"] = sites["event_id"].map(lambda e: bool(div.get(e, False)))
    # forecast uncertainty per site = 1 - mean forecast_confidence
    if len(fc):
        fu = fc.groupby("event_id")["forecast_confidence"].mean()
        sites["forecast_uncertainty"] = sites["event_id"].map(lambda e: 1.0 - float(fu.get(e, 0.5)))
    else:
        sites["forecast_uncertainty"] = 0.5

    # Step 4: spillover (first use of topology)
    spill = compute_spillover(sites, topo)
    sites["spillover_risk"] = sites["event_id"].map(lambda e: round(float(spill.get(e, 0.0)), 6))
    sites["spillover_norm"] = _norm(sites["spillover_risk"]).round(6)

    # Step 11: centrality
    centrality = build_centrality(topo) if len(topo) else pd.DataFrame()
    cmap = (dict(zip(centrality["site_id"].astype(str), centrality["weighted_degree_centrality"]))
            if len(centrality) else {})
    sites["topology_centrality"] = sites["event_id"].map(lambda e: float(cmap.get(e, 0.0)))
    sites["topology_centrality_norm"] = _norm(sites["topology_centrality"]).round(6)

    actions = action_dicts()
    score_rows, sim_rows, rec_rows, expl_rows = [], [], [], []

    for _, s in sites.iterrows():
        site = s.to_dict()
        j_base = baseline_J(site)
        best = None  # (control_score, action, sim, score)
        for a in actions:
            if not is_applicable(a, site):
                continue
            sim = simulate_action(site, a)
            sc = score_action(site, a, sim)
            score_rows.append({
                "event_id": site["event_id"], "action_id": a["action_id"],
                "action_type": a["action_type"], **sim, **sc,
                "feasible_applicable": True, "generated_at": _NOW_ISO})
            if best is None or sc["control_score"] > best[3]["control_score"]:
                best = (a["action_id"], a, sim, sc)
        # recommendation = best action if it nets positive benefit, else NO_ACTION
        if best is not None and best[3]["control_score"] > 0:
            aid, a, sim, sc = best
            rtype, rcount, approval = a["resource_type"], a["resource_count"], a["approval_required"]
            benefit, conf = sc["benefit_score"], 1.0 - site["forecast_uncertainty"]
        else:
            aid, a, sim = "NO_ACTION", None, simulate_action(site, None)
            rtype, rcount, approval, benefit = None, 0, False, 0.0
            conf = 1.0 - site["forecast_uncertainty"]

        wif = whatif(site, a if aid != "NO_ACTION" else None)
        sim_rows.append({"event_id": site["event_id"], "recommended_action": aid, **wif,
                         "generated_at": _NOW_ISO})
        rec_rows.append({
            "event_id": site["event_id"], "recommended_action": aid,
            "action_type": (a["action_type"] if a else "none"),
            "reason": expl.explain(site, aid, sim, site["spillover_risk"]),
            "expected_benefit": round(float(benefit), 6),
            "expected_queue_reduction_m": sim["d_queue"], "expected_travel_reduction_min": sim["d_travel"],
            "expected_risk_reduction": sim["d_risk"],
            "confidence": round(float(conf), 4),
            "resource_type": rtype if rtype else "none", "resource_count": int(rcount or 0),
            "resource_requirement": (f"{rtype}:{rcount}" if rtype else "none"),
            "risk_if_ignored": expl.risk_if_ignored(site),
            "approval_required": bool(approval) and aid not in INFORMATION_ONLY,
            "information_only": aid in INFORMATION_ONLY,
            "generated_at": _NOW_ISO})
        expl_rows.append({"event_id": site["event_id"], "recommended_action": aid,
                          "explanation": expl.explain(site, aid, sim, site["spillover_risk"]),
                          "risk_if_ignored": expl.risk_if_ignored(site), "generated_at": _NOW_ISO})

    recs = pd.DataFrame(rec_rows)
    scores = pd.DataFrame(score_rows)
    simulation = pd.DataFrame(sim_rows)
    explanations = pd.DataFrame(expl_rows)

    # Step 12: operator priority
    pr = sites[["event_id", "escalation_risk", "capacity_utilization",
                "spillover_norm", "topology_centrality_norm", "forecast_uncertainty"]].copy()
    pr["operator_priority_score"] = (
        0.35 * pr["escalation_risk"] + 0.25 * pr["capacity_utilization"]
        + 0.20 * pr["spillover_norm"] + 0.10 * pr["topology_centrality_norm"]
        + 0.10 * pr["forecast_uncertainty"]).clip(0, 1).round(6)
    p = pr["operator_priority_score"].rank(pct=True, method="average")
    pr["operator_priority_tier"] = np.where(p >= 0.90, "CRITICAL",
                                    np.where(p >= 0.70, "HIGH",
                                    np.where(p >= 0.40, "MEDIUM", "LOW")))
    pr["generated_at"] = _NOW_ISO

    # merge priority into recs for feasibility ordering
    recs = recs.merge(pr[["event_id", "operator_priority_score", "operator_priority_tier"]],
                      on="event_id", how="left")

    # Step 7: resource feasibility
    recs, constraints = apply_resource_feasibility(recs)

    # operator action queue (approval framework, Step 8)
    operator_actions = recs[[
        "event_id", "recommended_action", "action_type", "reason", "expected_benefit",
        "confidence", "resource_requirement", "risk_if_ignored", "approval_required",
        "information_only", "feasible", "resource_constrained",
        "operator_priority_score", "operator_priority_tier", "generated_at"]].copy()
    operator_actions["approval_status"] = np.where(
        operator_actions["information_only"], "auto_informational", "pending_approval")
    operator_actions = operator_actions.sort_values(
        "operator_priority_score", ascending=False).reset_index(drop=True)

    # diagnostics
    diag = _diagnostics(recs, scores, constraints, sites)

    if write:
        recs.to_csv(OUT / "layer7_control_recommendations.csv", index=False)
        scores.to_csv(OUT / "layer7_control_action_scores.csv", index=False)
        simulation.to_csv(OUT / "layer7_control_simulation.csv", index=False)
        constraints.to_csv(OUT / "layer7_control_constraints.csv", index=False)
        operator_actions.to_csv(OUT / "layer7_operator_actions.csv", index=False)
        pr.to_csv(OUT / "layer7_operator_priority.csv", index=False)
        explanations.to_csv(OUT / "layer7_control_explanations.csv", index=False)
        if len(centrality):
            centrality.to_csv(OUT / "layer7_topology_centrality.csv", index=False)
        diag.to_csv(OUT / "layer7_control_diagnostics.csv", index=False)

    checks = _validate(recs, scores, simulation, constraints, centrality, sites, pr, operator_actions)
    if write:
        _summary(recs, scores, constraints, pr, centrality, sites, checks)
    return {"recs": recs, "scores": scores, "simulation": simulation, "constraints": constraints,
            "operator_actions": operator_actions, "priority": pr, "explanations": explanations,
            "centrality": centrality, "diagnostics": diag, "sites": sites}, checks


def _diagnostics(recs, scores, constraints, sites) -> pd.DataFrame:
    rows = []

    def add(group, metric, value):
        rows.append({"group": group, "metric": metric, "value": value, "generated_at": _NOW_ISO})

    for k, v in recs["recommended_action"].value_counts().items():
        add("action_counts", str(k), int(v))
    add("feasibility", "feasible_rate",
        round(float(recs["feasible"].mean()), 4) if len(recs) else 0.0)
    add("feasibility", "resource_constrained", int(recs["resource_constrained"].sum()))
    for _, c in constraints.iterrows():
        add("resource_bottleneck", c["resource_type"],
            f"demand={c['demand']}/reserve={c['reserve_available']} bottleneck={c['bottleneck']}")
    act = recs[recs["recommended_action"] != "NO_ACTION"]
    add("benefit", "avg_expected_benefit", round(float(act["expected_benefit"].mean()), 6) if len(act) else 0.0)
    add("benefit", "avg_queue_reduction_m", round(float(act["expected_queue_reduction_m"].mean()), 4) if len(act) else 0.0)
    add("benefit", "avg_travel_reduction_min", round(float(act["expected_travel_reduction_min"].mean()), 4) if len(act) else 0.0)
    add("benefit", "avg_risk_reduction", round(float(act["expected_risk_reduction"].mean()), 6) if len(act) else 0.0)
    add("cost", "avg_cost_score", round(float(scores["cost_score"].mean()), 4) if len(scores) else 0.0)
    add("spillover", "mean_spillover", round(float(sites["spillover_risk"].mean()), 4))
    add("spillover", "max_spillover", round(float(sites["spillover_risk"].max()), 4))
    return pd.DataFrame(rows)


def _validate(recs, scores, simulation, constraints, centrality, sites, pr, operator_actions) -> list[dict]:
    checks = []

    def chk(cid, passed, detail, sev="critical"):
        checks.append({"check_id": cid, "phase": "control_engine", "passed": bool(passed),
                       "detail": detail, "severity": "info" if passed else sev})

    chk("m7c_scores_finite", bool(np.isfinite(scores[["benefit_score", "cost_score",
        "control_score", "objective_J"]].to_numpy()).all()) if len(scores) else True,
        f"{len(scores)} action scores finite")
    chk("m7c_recs_explainable", bool((recs["reason"].astype(str).str.len() > 0).all()),
        "every recommendation has a metric-grounded reason")
    chk("m7c_recs_reference_metrics",
        bool(recs["reason"].astype(str).str.contains(r"queue|escalation|capacity").all()),
        "explanations reference actual metrics")
    chk("m7c_resource_constraints_respected",
        bool((~constraints["bottleneck"]).all() or (constraints["allocated"] <= constraints["reserve_available"]).all()),
        f"reserve respected; demand={constraints['demand'].tolist()} reserve={constraints['reserve_available'].tolist()}")
    chk("m7c_spillover_finite", bool(np.isfinite(sites["spillover_risk"]).all()),
        f"spillover range [{sites['spillover_risk'].min():.3f},{sites['spillover_risk'].max():.3f}]")
    chk("m7c_centrality_finite",
        bool(len(centrality) == 0 or np.isfinite(centrality[["degree_centrality",
            "weighted_degree_centrality", "betweenness_centrality"]].to_numpy()).all()),
        f"{len(centrality)} nodes; centrality finite")
    chk("m7c_simulation_finite", bool(np.isfinite(simulation[["baseline_queue", "action_queue",
        "queue_impact", "risk_impact"]].to_numpy()).all()), "simulation outputs finite")
    # approval framework: non-information recs must require approval; nothing executed
    nonfo = operator_actions[~operator_actions["information_only"]]
    chk("m7c_approval_enforced",
        bool(nonfo["approval_required"].all()) and bool((operator_actions["approval_status"]
            .isin(["pending_approval", "auto_informational"]).all())),
        "all non-information recommendations require approval; none executed")
    chk("m7c_priority_bounded", bool(((pr["operator_priority_score"] >= 0)
        & (pr["operator_priority_score"] <= 1)).all()), "operator_priority in [0,1]")
    return checks


def _summary(recs, scores, constraints, pr, centrality, sites, checks) -> None:
    n_pass = sum(1 for c in checks if c["passed"]); n_fail = sum(1 for c in checks if not c["passed"])
    act = recs[recs["recommended_action"] != "NO_ACTION"]
    lines = [
        "LAYER 7 — M7C MPC-LITE CONTROL RECOMMENDATION SUMMARY",
        "=" * 54,
        f"generated_at: {_NOW_ISO}",
        "human-in-the-loop: every non-information recommendation requires approval; none executed.",
        "",
        f"A. recommendations generated: {len(recs)} (active actions: {len(act)}; NO_ACTION: {len(recs)-len(act)})",
        f"B. action-type distribution: {recs['recommended_action'].value_counts().to_dict()}",
        f"C. resource-constrained rate: {recs['resource_constrained'].mean():.3f} "
        f"({int(recs['resource_constrained'].sum())} actions)",
        f"D. avg expected queue reduction: {act['expected_queue_reduction_m'].mean():.1f} m" if len(act) else "D. avg queue reduction: n/a",
        f"E. avg expected travel-time reduction: {act['expected_travel_reduction_min'].mean():.2f} min" if len(act) else "E. n/a",
        f"F. avg escalation-risk reduction: {act['expected_risk_reduction'].mean():.4f}" if len(act) else "F. n/a",
        f"G. spillover-risk: mean={sites['spillover_risk'].mean():.3f} max={sites['spillover_risk'].max():.3f}",
        f"H. centrality: betweenness max={centrality['betweenness_centrality'].max():.4f} "
        f"weighted_degree max={centrality['weighted_degree_centrality'].max():.3f}" if len(centrality) else "H. n/a",
        f"I. operator-priority tiers: {pr['operator_priority_tier'].value_counts().to_dict()}",
        "J. future APIs: recommendations + operator_actions are flat, schema-stable feeds ready to serve.",
        "K. future GNN: spillover + centrality + per-node features are graph-ready signals.",
        "",
        f"VALIDATION: {n_pass} passed / {n_fail} failed",
    ]
    for c in checks:
        lines.append(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['check_id']}: {c['detail']}")
    (OUT / "layer7_control_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("LAYER 7 — M7C MPC-LITE CONTROL RECOMMENDATION ENGINE")
    print("=" * 60)
    tables, checks = run(write=True)
    r = tables["recs"]
    print(f"recommendations: {len(r)}  active: {int((r['recommended_action']!='NO_ACTION').sum())}  "
          f"resource_constrained: {int(r['resource_constrained'].sum())}")
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
    n_fail = sum(1 for c in checks if not c["passed"])
    print(f"\nValidation: {len(checks)-n_fail} passed / {n_fail} failed")


if __name__ == "__main__":
    main()
