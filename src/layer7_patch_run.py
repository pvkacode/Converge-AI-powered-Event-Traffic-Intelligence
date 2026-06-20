"""
Layer 7 — PATCH runner.

Regenerates the Layer 7 chain in dependency order after the audit fixes
(F-001, F-002, F-003, F-004, F-005, F-014 quality gate), schema-preserving, then
writes outputs/layer7_patch_validation.csv proving each fix achieved its goal.

Order: quality_gate FIRST, then the existing milestone orchestrators M1..M7 (which
invoke the patched modules). Layers 1-6 are never touched.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

import layer7_m1_run
import layer7_m2_run
import layer7_m3_run
import layer7_m4_run
import layer7_m5_run
import layer7_m6_run
import layer7_m7_run
import layer7_quality_gate as qg
from layer7_config import OUT

_NOW_ISO = datetime.now(timezone.utc).isoformat()


def _contrib_shares(state: pd.DataFrame) -> dict:
    cols = [c for c in state.columns if c.startswith("contrib_")]
    a = state[cols].abs().mean()
    tot = a.sum()
    return (a / tot).round(4).to_dict() if tot > 0 else {}


def build_patch_validation() -> pd.DataFrame:
    rows: list[dict] = []

    def chk(fid, check, passed, detail):
        rows.append({"finding": fid, "check": check, "passed": bool(passed),
                     "detail": detail, "generated_at": _NOW_ISO})

    # --- Quality gate (F-014)
    g = pd.read_csv(OUT / "layer7_quality_gate.csv")
    chk("F-014", "quality_gate_built", len(g) > 0 and g["quality_weight"].between(0.6, 1.0).all(),
        f"{len(g)} events; weight in [{g.quality_weight.min():.2f},{g.quality_weight.max():.2f}]; "
        f"flagged={int((g.data_quality_flag=='flagged').sum())}")
    st = pd.read_csv(OUT / "layer7_operational_state.csv")
    chk("F-014", "quality_factor_applied", "quality_factor" in st.columns and "data_quality_flag" in st.columns,
        "operational_state carries quality_factor + data_quality_flag (ORS down-weighted)")

    # --- F-001 ORS dominance
    shares = _contrib_shares(st)
    maxshare = max(shares.values()) if shares else 1.0
    chk("F-001", "no_single_feature_dominance", maxshare < 0.50,
        f"max |contribution| share = {maxshare:.3f} (was 0.94 tail); shares={shares}")
    chk("F-001", "ors_schema_preserved",
        all(c in st.columns for c in ["operational_risk_score", "operational_tier", "ors_percentile"]),
        "ORS columns intact")
    chk("F-001", "ors_no_nan", int(st["operational_risk_score"].isna().sum()) == 0,
        f"{int(st['operational_risk_score'].isna().sum())} NaN ORS")

    # --- F-002 ASS bounded
    al = pd.read_csv(OUT / "layer7_prioritized_alerts.csv")
    amax = float(al["alert_severity_score"].max())
    chk("F-002", "ass_bounded_0_1", amax <= 1.0 + 1e-9,
        f"max ASS = {amax:.4f} (was 1.262); raw retained in alert_severity_raw={'alert_severity_raw' in al.columns}")
    chk("F-002", "ass_schema_preserved",
        all(c in al.columns for c in ["alert_severity_score", "priority"]),
        "ASS + priority columns intact")

    # --- F-003 P1 inflation
    p1 = float((al["priority"] == "P1").mean()) if len(al) else 0.0
    chk("F-003", "p1_rate_reduced", p1 <= 0.20 + 1e-9,
        f"P1 share = {p1:.3f} (was 0.315); dist={al['priority'].value_counts().to_dict()}")

    # --- F-004 OIS band reachable
    ext = pd.read_csv(OUT / "layer7_override_impact_extended.csv")
    levels = set(ext["impact_level"].unique())
    hi = int(ext["impact_level"].isin(["High", "Critical"]).sum())
    chk("F-004", "impact_band_reachable", hi >= 1 and len({"High", "Critical"} & levels) >= 1,
        f"High/Critical overrides = {hi}; impact_level set={sorted(levels)} (was only Low/Moderate)")
    gov = pd.read_csv(OUT / "layer7_governance_summary.csv").set_index("metric")["value"].to_dict()
    chk("F-004", "high_impact_rate_meaningful", float(gov.get("high_impact_override_rate", 0)) > 0,
        f"high_impact_override_rate = {gov.get('high_impact_override_rate')} (was always 0.0)")

    # --- F-005 TCS double-count removed
    tc = pd.read_csv(OUT / "layer7_twin_confidence.csv")
    has_stab = "scenario_stability" in tc.columns
    # verify TCS == 0.5*dcs + 0.5*stability (no robustness/uncertainty in the sum)
    recon = (0.5 * tc["dcs_component"] + 0.5 * tc["scenario_stability"]).clip(0, 1)
    matches = bool(np.allclose(recon, tc["twin_confidence_score"], atol=1e-6)) if has_stab else False
    chk("F-005", "tcs_independent_axis", has_stab and matches,
        f"TCS = 0.5*DCS + 0.5*scenario_stability (entropy 1-H/logK); robustness/uncertainty now diagnostic-only; "
        f"TCS range [{tc.twin_confidence_score.min():.3f},{tc.twin_confidence_score.max():.3f}]")
    chk("F-005", "tcs_schema_preserved",
        all(c in tc.columns for c in ["twin_confidence_score", "dcs_component", "tcs_tier"]),
        "TCS columns intact + scenario_stability added")

    # --- no-NaN on the PATCHED/NEW columns only (pre-existing structural NaN in
    # robustness_score/service_tier/affected_event_id are preserved by design).
    patched_cols = {
        "layer7_quality_gate.csv": ["quality_weight", "data_quality_flag"],
        "layer7_operational_state.csv": ["operational_risk_score", "quality_factor",
                                         "data_quality_flag", "ors_percentile"],
        "layer7_prioritized_alerts.csv": ["alert_severity_score", "alert_severity_raw", "priority"],
        "layer7_override_impact_extended.csv": ["relative_ois", "absolute_ois", "impact_level"],
        "layer7_twin_confidence.csv": ["twin_confidence_score", "dcs_component", "scenario_stability"],
    }
    nan_total = 0
    for f, cols in patched_cols.items():
        d = pd.read_csv(OUT / f)
        nan_total += int(d[[c for c in cols if c in d.columns]].isna().sum().sum())
    chk("ALL", "no_nan_patched_columns", nan_total == 0,
        f"{nan_total} NaN in patched/new columns (pre-existing structural NaN preserved by design)")

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 64)
    print("LAYER 7 — PATCH RUN (F-001/2/3/4/5 + L6 integrity gate)")
    print("=" * 64)
    qg.run(write=True)                 # quality gate FIRST
    for m in (layer7_m1_run, layer7_m2_run, layer7_m3_run, layer7_m4_run,
              layer7_m5_run, layer7_m6_run, layer7_m7_run):
        m.main()
    report = build_patch_validation()
    report.to_csv(OUT / "layer7_patch_validation.csv", index=False)
    n_pass = int(report["passed"].sum()); n_fail = int((~report["passed"]).sum())
    print("\n" + "=" * 64)
    print("PATCH VALIDATION")
    for _, r in report.iterrows():
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['finding']}/{r['check']}: {r['detail']}")
    print(f"\n{n_pass} passed / {n_fail} failed.")


if __name__ == "__main__":
    main()
