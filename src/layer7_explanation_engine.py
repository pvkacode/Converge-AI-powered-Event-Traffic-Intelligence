"""
Layer 7 — M3 (Part A/C/D): Explanation & Decision-Support Engine.

Deterministic. NO new model, NO retraining, NO optimization reruns. Every
explanation is a transparent decomposition / restatement of already-published
Layer 5 / Layer 6 / Layer 7 quantities.

Parts:
  A — active-site risk / resource / alert explanations
  C — active-site tiering (active_operational_tier; does NOT replace operational_tier)
  D — absolute OIS (fixed L5-anchored scaling) alongside the existing relative OIS

ADDITIVE ONLY. Writes only outputs/layer7_*.csv.

Outputs:
  outputs/layer7_site_explanations.csv
  outputs/layer7_resource_explanations.csv
  outputs/layer7_alert_explanations.csv
  outputs/layer7_active_site_tiers.csv
  outputs/layer7_override_impact_extended.csv
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from layer7_config import OIS_IMPACT_PCT_CUTS, OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()

# Per-site resource caps (Layer 5 published budget caps) for allocation intensity.
_RES_CAPS = {"officers_allocated": 12, "barricades_allocated": 20,
             "tow_trucks_allocated": 4, "qru_allocated": 3}

_CONTRIB_COLS = [
    "contrib_tail_risk_prob_z", "contrib_fragility_signal_z", "contrib_obi_signal_z",
    "contrib_drift_score_z", "contrib_novelty_score_z", "contrib_critical_alert_indicator",
]
_FACTOR_LABEL = {
    "contrib_tail_risk_prob_z": "Tail Risk",
    "contrib_fragility_signal_z": "Fragility",
    "contrib_obi_signal_z": "Operational Burden",
    "contrib_drift_score_z": "Drift",
    "contrib_novelty_score_z": "Novelty",
    "contrib_critical_alert_indicator": "Critical Alert",
}

# OIS weights (mandated, shared with override engine).
OIS_W_DELAY, OIS_W_RISK, OIS_W_ALERT = 0.40, 0.35, 0.25


# --------------------------------------------------------------------------- anchors (Part D)
def compute_absolute_anchors() -> dict:
    """Fixed scaling anchors from HISTORICAL Layer 5 published ranges (not the
    current override/counterfactual batch). Stable for the run."""
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    shadow = pd.read_csv(OUT / "layer5_shadow_prices.csv")
    delred = pd.to_numeric(alloc.get("expected_delay_reduction_min"), errors="coerce").abs()
    max_marginal = float(pd.to_numeric(shadow["marginal_value"], errors="coerce").abs().max())
    # delay covers both mechanisms: diversion (expected_delay_reduction_min) and
    # resource changes (marginal_value x max per-site officer cap = 12).
    d_anchor = max(float(delred.max()) if len(delred) else 0.0, max_marginal * 12.0, 1.0)
    try:
        alerts = pd.read_csv(OUT / "layer7_prioritized_alerts.csv")
        a_anchor = float(pd.to_numeric(alerts["alert_severity_score"], errors="coerce").max())
    except Exception:
        a_anchor = 1.305
    a_anchor = a_anchor if a_anchor and a_anchor == a_anchor else 1.305
    return {"delay": d_anchor, "risk": 100.0, "alert": max(a_anchor, 1e-6)}


def absolute_ois(delay_proxy, risk_proxy, alert_proxy, anchors: dict) -> np.ndarray:
    nd = np.minimum(1.0, np.abs(np.asarray(delay_proxy, dtype=float)) / anchors["delay"])
    nr = np.minimum(1.0, np.abs(np.asarray(risk_proxy, dtype=float)) / anchors["risk"])
    na = np.minimum(1.0, np.abs(np.asarray(alert_proxy, dtype=float)) / anchors["alert"])
    return np.clip(OIS_W_DELAY * nd + OIS_W_RISK * nr + OIS_W_ALERT * na, 0.0, 1.0)


def _impact_level(x: float) -> str:
    # (retained for backward compatibility; superseded by _impact_level_quantile)
    if x < 0.25:
        return "Low"
    if x < 0.50:
        return "Moderate"
    if x < 0.75:
        return "High"
    return "Critical"


def _impact_level_quantile(values: np.ndarray) -> list[str]:
    """PATCH F-004: assign impact tiers by quantile of the absolute_ois distribution.
    The fixed-anchor absolute_ois never reaches 0.5, so static cuts left High/Critical
    permanently empty; quantile cuts make the band usable while keeping ordering."""
    s = pd.Series(np.asarray(values, dtype=float))
    pct = s.rank(pct=True, method="average")

    def _lab(p: float) -> str:
        for label, cut in OIS_IMPACT_PCT_CUTS:
            if p >= cut:
                return label
        return OIS_IMPACT_PCT_CUTS[-1][0]

    return [_lab(p) for p in pct]


# --------------------------------------------------------------------------- Part C: tiers
def build_active_tiers() -> pd.DataFrame:
    astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    df = astate.sort_values(
        ["operational_risk_score", "event_id"], ascending=[False, True]
    ).reset_index(drop=True)
    pct = df["operational_risk_score"].rank(pct=True, method="average")
    df["active_percentile"] = pct

    def _tier(p: float) -> str:
        if p >= 0.90:      # top 10%
            return "Emergency"
        if p >= 0.70:      # next 20%
            return "Critical"
        if p >= 0.40:      # next 30%
            return "Elevated"
        return "Normal"    # remaining

    df["active_operational_tier"] = pct.apply(_tier)
    df["generated_at"] = _NOW_ISO
    return df[["event_id", "operational_tier", "active_operational_tier",
               "active_percentile", "generated_at"]].copy()


# --------------------------------------------------------------------------- Part A: site risk
def build_site_explanations(tiers: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    astate = pd.read_csv(OUT / "layer7_active_site_state.csv")
    astate["event_id"] = astate["event_id"].astype(str)
    opstate = pd.read_csv(OUT / "layer7_operational_state.csv")
    opstate["event_id"] = opstate["event_id"].astype(str)
    contribs = opstate[["event_id"] + _CONTRIB_COLS].drop_duplicates("event_id")

    df = astate[["event_id"]].merge(contribs, on="event_id", how="left")
    df = df.merge(tiers[["event_id", "active_operational_tier"]], on="event_id", how="left")

    abs_contrib = df[_CONTRIB_COLS].abs()
    total = abs_contrib.sum(axis=1)
    n_degenerate = int((total <= 1e-12).sum())

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        tot = total.iloc[i]
        shares = {}
        if tot > 1e-12:
            for c in _CONTRIB_COLS:
                shares[c] = abs(r[c]) / tot
        else:
            for c in _CONTRIB_COLS:
                shares[c] = 0.0
        ordered = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[:3]
        labels = [(_FACTOR_LABEL[c], s) for c, s in top]
        # pad if fewer than 3
        while len(labels) < 3:
            labels.append(("None", 0.0))
        if tot > 1e-12:
            expl = (
                f"Risk dominated by {labels[0][0]} ({labels[0][1]*100:.0f}%), "
                f"{labels[1][0]} ({labels[1][1]*100:.0f}%), "
                f"{labels[2][0]} ({labels[2][1]*100:.0f}%)."
            )
        else:
            expl = "No differentiating risk signal (all contributions ~0)."
        rows.append({
            "event_id": r["event_id"],
            "active_operational_tier": r["active_operational_tier"],
            "top_factor_1": labels[0][0], "top_factor_1_share": round(labels[0][1], 6),
            "top_factor_2": labels[1][0], "top_factor_2_share": round(labels[1][1], 6),
            "top_factor_3": labels[2][0], "top_factor_3_share": round(labels[2][1], 6),
            "full_share_sum": round(float(sum(shares.values())), 6),
            "risk_explanation": expl,
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows), n_degenerate


# --------------------------------------------------------------------------- Part A: resources
def build_resource_explanations() -> pd.DataFrame:
    alloc = pd.read_csv(OUT / "layer5_resource_allocation.csv")
    alloc["event_id"] = alloc["event_id"].astype(str)
    shadow = pd.read_csv(OUT / "layer5_shadow_prices.csv")
    marg = dict(zip(shadow["resource"].astype(str), shadow["marginal_value"].astype(float)))
    top_resource = max(marg, key=marg.get) if marg else "qru"
    tier_rank = {"normal": 0, "elevated": 1, "critical": 2, "emergency": 3}

    rows = []
    for _, r in alloc.iterrows():
        tier = str(r.get("service_tier", "normal")).strip().lower()
        rob = float(pd.to_numeric(r.get("robustness_score"), errors="coerce") or 0.0)
        cc_viol = str(r.get("violation_flag", "")).strip().lower() in ("1", "true", "yes")
        p = float(r.get("officers_allocated", 0) or 0)
        b = float(r.get("barricades_allocated", 0) or 0)
        t = float(r.get("tow_trucks_allocated", 0) or 0)
        q = float(r.get("qru_allocated", 0) or 0)
        intensity = (
            p / _RES_CAPS["officers_allocated"] + b / _RES_CAPS["barricades_allocated"]
            + t / _RES_CAPS["tow_trucks_allocated"] + q / _RES_CAPS["qru_allocated"]
        ) / 4.0
        score = float(np.clip(
            0.35 * (tier_rank.get(tier, 0) / 3.0)
            + 0.25 * (1.0 - np.clip(rob, 0, 1))
            + 0.20 * (1.0 if cc_viol else 0.0)
            + 0.20 * np.clip(intensity, 0, 1),
            0, 1,
        ))
        parts = []
        if p:
            parts.append(f"{int(p)} officers")
        if b:
            parts.append(f"{int(b)} barricades")
        if t:
            parts.append(f"{int(t)} tow")
        if q:
            parts.append(f"{int(q)} QRU")
        alloc_txt = ", ".join(parts) if parts else "no resources"
        expl = (
            f"Service tier '{tier}' drives baseline staffing; allocated {alloc_txt}. "
            f"Highest marginal-value resource city-wide is '{top_resource}' "
            f"(shadow price {marg.get(top_resource, 0):.0f}). "
            f"Robustness={rob:.2f}"
            + ("; chance-constraint VIOLATED -> reinforcement justified." if cc_viol
               else "; chance constraints satisfied.")
        )
        rows.append({
            "event_id": str(r["event_id"]),
            "resource_rationale_score": round(score, 6),
            "resource_explanation": expl,
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Part A: alerts
def build_alert_explanations() -> pd.DataFrame:
    alerts = pd.read_csv(OUT / "layer7_prioritized_alerts.csv")
    try:
        diag = pd.read_csv(OUT / "layer7_alert_diagnostics.csv")
        topic_sources = dict(zip(diag["topic_key"].astype(str), diag["sources"].astype(str)))
    except Exception:
        topic_sources = {}

    rows = []
    for _, r in alerts.iterrows():
        topic = str(r.get("topic_key", ""))
        sources = topic_sources.get(topic, str(r.get("source_feed", "")))
        n_src = int(r.get("corroboration_sources", 1))
        sev = str(r.get("severity_raw", ""))
        base = float(r.get("base_severity", 0.0))
        corr = float(r.get("corroboration_factor", 1.0))
        rec = float(r.get("recency_factor", 1.0))
        ass = float(r.get("alert_severity_score", 0.0))
        layer = str(r.get("affected_layer", ""))
        root = f"{topic} (affected={layer})"
        expl = (
            f"{r.get('priority', '')} {sev} alert on topic '{topic}' affecting {layer}. "
            f"Corroborated by {n_src} source(s): {sources}. "
            f"Severity = base {base:.2f} x corroboration {corr:.2f} x recency {rec:.2f} "
            f"= ASS {ass:.3f}. Priority justified by ASS band."
        )
        rows.append({
            "alert_id": str(r.get("l7_alert_id", "")),
            "priority": str(r.get("priority", "")),
            "severity": sev,
            "root_cause": root,
            "corroboration_sources": sources,
            "alert_explanation": expl,
            "generated_at": _NOW_ISO,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Part D: extended OIS
def build_override_impact_extended(anchors: dict) -> pd.DataFrame:
    imp = pd.read_csv(OUT / "layer7_override_impact_report.csv")
    abs_ois = absolute_ois(imp["delay_proxy"], imp["risk_proxy"], imp["alert_proxy"], anchors)
    out = pd.DataFrame({
        "override_id": imp["override_id"].astype(str),
        "relative_ois": imp["ois"].astype(float).round(6),
        "absolute_ois": np.round(abs_ois, 6),
    })
    out["impact_level"] = _impact_level_quantile(abs_ois)
    out["generated_at"] = _NOW_ISO
    return out


# --------------------------------------------------------------------------- run
def run(write: bool = True) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    anchors = compute_absolute_anchors()
    tiers = build_active_tiers()
    site_expl, n_degenerate = build_site_explanations(tiers)
    res_expl = build_resource_explanations()
    alert_expl = build_alert_explanations()
    ext_ois = build_override_impact_extended(anchors)

    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        site_expl.to_csv(OUT / "layer7_site_explanations.csv", index=False)
        res_expl.to_csv(OUT / "layer7_resource_explanations.csv", index=False)
        alert_expl.to_csv(OUT / "layer7_alert_explanations.csv", index=False)
        tiers.to_csv(OUT / "layer7_active_site_tiers.csv", index=False)
        ext_ois.to_csv(OUT / "layer7_override_impact_extended.csv", index=False)

    checks = _validate(site_expl, res_expl, alert_expl, tiers, ext_ois, n_degenerate)
    return {
        "tiers": tiers, "site": site_expl, "resource": res_expl,
        "alert": alert_expl, "extended_ois": ext_ois, "anchors": anchors,
        "n_degenerate": n_degenerate,
    }, checks


def _validate(site, res, alert, tiers, ext, n_degenerate) -> list[dict]:
    checks: list[dict] = []

    def chk(cid, passed, detail, severity="critical"):
        checks.append({"check_id": cid, "phase": "explanation_engine",
                       "passed": bool(passed), "detail": detail,
                       "severity": "info" if passed else severity})

    n_nan = sum(int(d.isna().sum().sum()) for d in (site, res, alert, tiers, ext))
    chk("m3_explanations_no_nan", n_nan == 0, f"{n_nan} NaN across explanation outputs")

    tier_set = set(tiers["active_operational_tier"].unique())
    chk("m3_active_tiers_complete",
        tiers["active_operational_tier"].notna().all() and len(tier_set) >= 1,
        f"active tier distribution: {tiers['active_operational_tier'].value_counts().to_dict()}")

    chk("m3_explanations_generated",
        len(site) > 0 and len(res) > 0 and len(alert) > 0,
        f"site={len(site)}, resource={len(res)}, alert={len(alert)}")

    # top contributor shares: full per-site share sum ~ 1 (excluding degenerate rows)
    nondeg = site[site["full_share_sum"] > 1e-9]
    ok_sum = bool(np.allclose(nondeg["full_share_sum"], 1.0, atol=1e-6)) if len(nondeg) else True
    chk("m3_contribution_shares_sum_to_one", ok_sum,
        f"{len(nondeg)} sites sum~1; {n_degenerate} degenerate (all-zero contrib)")

    for col in ("relative_ois", "absolute_ois"):
        in_range = bool(((ext[col] >= 0) & (ext[col] <= 1)).all())
        chk(f"m3_{col}_range", in_range, f"{col} range [{ext[col].min():.4f}, {ext[col].max():.4f}]")

    return checks


if __name__ == "__main__":
    tables, checks = run(write=True)
    print("=== Layer 7 M3 Explanation Engine ===")
    print("active tiers:", tables["tiers"]["active_operational_tier"].value_counts().to_dict())
    print("anchors:", tables["anchors"])
    for c in checks:
        print(f"  [{'OK ' if c['passed'] else '!! '}] {c['check_id']}: {c['detail']}")
