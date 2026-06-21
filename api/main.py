"""
Converge / ASTraM thin inference API.

This is a NEW, separate service. It does not modify, import-for-side-effect, or
re-run any pipeline script that would write to outputs/ or data/. It:

  * Calls the REAL Layer 1 survival functions from src/layer1_survival.py to
    genuinely recompute duration quantiles per request (when the pipeline
    environment + cleaned dataset are available). This module is import-safe:
    it only defines functions/constants, guarded by `if __name__ == "__main__"`.
  * Serves every other layer from the precomputed CSV exports in outputs/
    (read-only), because those source modules (layer2/3/4/4.5/5/7) execute their
    full pipeline AND write to outputs/ at import time (no __main__ guard), so
    importing them would violate the read-only constraint, and Layer 5 (MILP) /
    Layer 7 (Hawkes) are batch-only and far too slow for a live request.

Every layer in the response carries a `provenance` flag:
    "live"               - a real model function ran this request
    "precomputed_lookup" - served from the existing outputs/ CSVs, keyed by input
    "fallback"           - the preferred path was unavailable; explained in `note`
"""
from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(os.environ.get("ASTRAM_ROOT", Path(__file__).resolve().parent.parent))
SRC = ROOT / "src"
OUT = Path(os.environ.get("OUTPUTS_DIR", ROOT / "outputs"))
FRONTEND = OUT / "frontend"
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data"))

_cors_raw = os.environ.get("CORS_ORIGINS", "*").strip()
CORS_ORIGINS = ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]

app = FastAPI(title="Converge / ASTraM Inference API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# CSV cache (read-only)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def csv(rel: str) -> Optional[pd.DataFrame]:
    p = OUT / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/inf with None so the JSON is valid."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Layer 1 LIVE engine (genuine survival model calls)
# ---------------------------------------------------------------------------
class Layer1Engine:
    """Builds Kaplan-Meier strata once from the real data, then answers
    per-request via src.layer1_survival.lookup_expected_duration."""

    def __init__(self) -> None:
        self.ok = False
        self.reason = ""
        self.l1 = None
        self.km_primary = None
        self.km_fallback = None
        try:
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            import layer1_survival as l1  # import-safe (functions + __main__ guard)

            self.l1 = l1
            df = l1.load_data()                       # reads data/events_clean.parquet
            surv = l1.build_survival_table(df)        # pure transform, no I/O
            self.km_primary = l1.fit_km_strata(surv, ["event_cause", "corridor"])
            self.km_fallback = l1.fit_km_strata(surv, ["event_cause"])
            self.ok = True
        except Exception as exc:  # missing deps / missing parquet / etc.
            self.reason = f"{type(exc).__name__}: {exc}"

    def lookup(self, cause: str, corridor: str) -> Optional[dict]:
        if not self.ok:
            return None
        out: dict[str, Any] = {}
        src = None
        n = None
        conf = None
        for q in ("p50", "p80", "p95"):
            r = self.l1.lookup_expected_duration(cause, corridor, self.km_primary, self.km_fallback, q)
            if r is None:
                return None
            out[q] = _f(r["duration_min"])
            src, n, conf = r["source"], r.get("n"), r.get("confidence")
        out["source"] = src
        out["n"] = int(n) if n is not None else None
        out["confidence"] = conf
        return out


_engine: Optional[Layer1Engine] = None


def engine() -> Layer1Engine:
    global _engine
    if _engine is None:
        _engine = Layer1Engine()
    return _engine


# ---------------------------------------------------------------------------
# Quantile-based risk tier (mirrors the dashboard's derived tier)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _risk_thresholds() -> Optional[tuple[float, float, float]]:
    df = csv("frontend/risk_scores.csv")
    if df is None or "survival_risk_score" not in df.columns:
        return None
    s = pd.to_numeric(df["survival_risk_score"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.quantile(0.5)), float(s.quantile(0.8)), float(s.quantile(0.95))


def _tier(score: Optional[float]) -> Optional[str]:
    th = _risk_thresholds()
    if score is None or th is None:
        return None
    t50, t80, t95 = th
    if score >= t95:
        return "Critical"
    if score >= t80:
        return "High"
    if score >= t50:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Duration sanity (Kaplan-Meier tails often clamp to last censored follow-up)
# ---------------------------------------------------------------------------
MAX_PLAUSIBLE_DURATION_MIN = 7 * 24 * 60  # one week
TAIL_INFLATION_RATIO = 15.0               # p80 > 15× p50 and > 4 h → suspect

# Layer 4 retrieval is trained on true planned-event causes only.
PLANNED_CAUSES = frozenset({"public_event", "procession", "protest", "vip_movement"})
NON_CORRIDOR = frozenset({"", "non-corridor"})


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_non_corridor(corridor: str) -> bool:
    return _norm(corridor) in NON_CORRIDOR


def _match_col(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().str.lower() == _norm(value)


def _first_positive(*values: Any) -> Optional[float]:
    for v in values:
        f = _f(v)
        if f is not None and f > 0:
            return f
    for v in values:
        f = _f(v)
        if f is not None:
            return f
    return None


def _resources_from_row(r: pd.Series) -> dict[str, Optional[float]]:
    """Prefer LP-allocated counts when positive; fall back to ODS manpower estimates."""
    return {
        "officers": _first_positive(r.get("allocated_officers"), r.get("officers")),
        "barricades": _first_positive(
            r.get("allocated_barricades"), r.get("barricades_from_ods"), r.get("barricades")
        ),
        "tow": _first_positive(r.get("allocated_tow"), r.get("tow_vehicles"), r.get("tow_unit")),
        "supervisors": _first_positive(r.get("allocated_supervisors"), r.get("supervisors")),
    }


def _resources_all_zero(res: dict[str, Optional[float]]) -> bool:
    return all((res.get(k) or 0) == 0 for k in ("officers", "barricades", "tow"))


def _quantile_sane(p50: Optional[float], p80: Optional[float], p95: Optional[float]) -> bool:
    """Return False when KM tail quantiles are censored or operationally implausible."""
    if p50 is not None and p50 > MAX_PLAUSIBLE_DURATION_MIN:
        return False
    for tail in (p80, p95):
        if tail is None:
            continue
        if tail > MAX_PLAUSIBLE_DURATION_MIN:
            return False
        if p50 is not None and p50 > 0 and tail > max(240.0, TAIL_INFLATION_RATIO * p50):
            return False
    if (
        p80 is not None
        and p95 is not None
        and abs(p80 - p95) < 1.0
        and max(p80, p95) > 1440.0
    ):
        return False
    return True


def _minutes_plausible(mins: Optional[float]) -> bool:
    return mins is not None and mins <= MAX_PLAUSIBLE_DURATION_MIN


def _fmt_duration_min(mins: float) -> str:
    if mins < 120:
        return f"{mins:.0f} min"
    if mins < 48 * 60:
        return f"{mins / 60:.1f} h"
    return f"{mins / 1440:.1f} d"


def _layer45_quantiles(section: dict) -> dict[str, Optional[float]]:
    dq = section.get("duration_quantiles") or {}
    return {
        "p50": _f(dq.get("p50")),
        "p80": _f(dq.get("p80")),
        "p95": _f(dq.get("p95")),
    }


def _resolve_duration_plan(sections: dict) -> Optional[dict[str, Any]]:
    """Pick operational planning quantiles; prefer guarded L4.5 when L1 tails are censored."""
    l1 = sections["layer1_duration"]
    l45 = sections.get("layer45_fusion") or {}
    guarded = _layer45_quantiles(l45)
    p50, p80, p95 = l1.get("p50"), l1.get("p80"), l1.get("p95")
    tails_ok = _quantile_sane(p50, p80, p95)

    if guarded.get("p80") is not None and (not tails_ok or p80 is None):
        return {
            "quantile": "p80",
            "minutes": guarded["p80"],
            "source": "layer45_guarded",
            "note": (
                "Layer 1 Kaplan-Meier tail quantiles are censored or implausible for this "
                "cause/corridor; planning uses Layer 4.5 guarded P80."
            ),
            "p50": guarded.get("p50") or (p50 if _minutes_plausible(p50) else None),
            "p80": guarded["p80"],
            "p95": guarded.get("p95"),
        }
    if p80 is not None and tails_ok:
        return {
            "quantile": "p80",
            "minutes": p80,
            "source": "layer1",
            "note": "Planning uses Layer 1 Kaplan-Meier P80.",
            "p50": p50,
            "p80": p80,
            "p95": p95,
        }
    if guarded.get("p50") is not None:
        return {
            "quantile": "p50",
            "minutes": guarded["p50"],
            "source": "layer45_guarded",
            "note": "No trustworthy P80; planning uses Layer 4.5 guarded P50.",
            "p50": guarded["p50"],
            "p80": guarded.get("p80"),
            "p95": guarded.get("p95"),
        }
    if _minutes_plausible(p50):
        return {
            "quantile": "p50",
            "minutes": p50,
            "source": "layer1",
            "note": "No trustworthy P80; planning uses Layer 1 P50 only.",
            "p50": p50,
            "p80": None,
            "p95": None,
        }
    return None


def _layer1_payload(
    *,
    provenance: str,
    p50: Optional[float],
    p80: Optional[float],
    p95: Optional[float],
    source: Optional[str],
    n: Optional[int],
    confidence: Optional[str],
    note: str,
    trustworthy_flag: bool,
) -> dict[str, Any]:
    tail_trustworthy = _quantile_sane(p50, p80, p95)
    out: dict[str, Any] = {
        "provenance": provenance,
        "p50": p50,
        "p80": p80,
        "p95": p95,
        "source": source,
        "n": n,
        "confidence": confidence,
        "trustworthy_flag": trustworthy_flag,
        "tail_trustworthy": tail_trustworthy,
        "note": note,
    }
    if not tail_trustworthy:
        out["tail_note"] = (
            "P80/P95 are right-censored Kaplan-Meier estimates (survival never reached the "
            "target probability). Do not use them for operational planning — the synthesised "
            "recommendation falls back to Layer 4.5 guarded quantiles when available."
        )
    return out


def _resolve_officer_count(sections: dict) -> tuple[Optional[float], Optional[str]]:
    """Prefer positive L3 junction allocation; else L4 retrieval; else L5 MILP export."""
    l3 = sections.get("layer3_resources") or {}
    l4 = sections.get("layer4_event") or {}
    l5 = sections.get("layer5_optimization") or {}

    l3_off = _f(l3.get("officers"))
    if l3_off is not None and l3_off > 0:
        return l3_off, "layer3"

    l4_off = _f((l4.get("recommended") or {}).get("officers"))
    if l4_off is not None and l4_off > 0:
        return l4_off, "layer4"

    l5_off = _f((l5.get("allocation") or {}).get("officers"))
    if l5_off is not None and l5_off > 0:
        return l5_off, "layer5"

    return None, None


def _junction_for_corridor(corridor: str, cause: Optional[str] = None) -> Optional[str]:
    """Pick the highest-burden junction on a corridor (optionally cause-filtered).

    For Non-corridor, prefer junctions with non-zero ODS manpower on the selected
    cause rather than the single highest-OBI junction (which may have LP alloc=0).
    """
    rs = csv("frontend/risk_scores.csv")
    ob = csv("frontend/operational_burden.csv")
    mp = csv("layer3_manpower_recommendations.csv")
    if rs is None:
        return None

    sub = rs[_match_col(rs, "corridor", corridor) & rs["junction"].notna() & (rs["junction"].astype(str) != "")]
    if cause and not sub.empty:
        cause_sub = sub[_match_col(sub, "event_cause", cause)]
        if not cause_sub.empty:
            sub = cause_sub

    if sub.empty:
        return None

    juncs = sub["junction"].astype(str).unique().tolist()

    # Prefer junctions with positive ODS manpower (avoids allocated_officers=0 trap).
    if mp is not None and "junction" in mp.columns:
        mp_sub = mp[mp["junction"].astype(str).isin(juncs)].copy()
        if not mp_sub.empty:
            mp_sub["officers"] = pd.to_numeric(mp_sub.get("officers"), errors="coerce").fillna(0)
            mp_sub["ods_score"] = pd.to_numeric(mp_sub.get("ods_score"), errors="coerce").fillna(0)
            nonzero = mp_sub[mp_sub["officers"] > 0]
            pick_from = nonzero if not nonzero.empty else mp_sub
            return str(pick_from.sort_values(["ods_score", "officers"], ascending=False).iloc[0]["junction"])

    if ob is not None and "operational_burden_index" in ob.columns:
        m = ob[ob["junction"].astype(str).isin(juncs)].copy()
        if not m.empty:
            m["operational_burden_index"] = pd.to_numeric(m["operational_burden_index"], errors="coerce")
            return str(m.sort_values("operational_burden_index", ascending=False).iloc[0]["junction"])

    return juncs[0]


# ---------------------------------------------------------------------------
# Per-layer assembly
# ---------------------------------------------------------------------------
def layer1_section(cause: str, corridor: str) -> dict:
    eng = engine()
    live = eng.lookup(cause, corridor) if eng.ok else None
    if live is not None:
        trustworthy = live.get("confidence") in ("high", "moderate")
        return _layer1_payload(
            provenance="live",
            p50=live["p50"],
            p80=live["p80"],
            p95=live["p95"],
            source=live["source"],
            n=live["n"],
            confidence=live["confidence"],
            trustworthy_flag=trustworthy,
            note="Kaplan-Meier survival quantiles recomputed live from data/events_clean.parquet.",
        )
    # ---- fallback to precomputed duration_lookup.csv ----
    dl = csv("frontend/duration_lookup.csv")
    reason = eng.reason or "live engine returned no match"
    if dl is not None:
        m = dl[(dl["event_cause"] == cause) & (dl["corridor"] == corridor)]
        src = "cause_corridor"
        if m.empty:
            # cause-only fallback row is not in this export; report honestly
            m = dl[dl["event_cause"] == cause]
            src = "cause_only_fallback"
        if not m.empty:
            row = m.iloc[0]
            n = _f(row.get("n"))
            return _layer1_payload(
                provenance="fallback",
                p50=_f(row.get("p50_min")),
                p80=_f(row.get("p80_min")),
                p95=_f(row.get("p95_min")),
                source=src,
                n=int(n) if n is not None else None,
                confidence="moderate",
                trustworthy_flag=True,
                note=f"Live survival engine unavailable ({reason}). Served from duration_lookup.csv.",
            )
    return _layer1_payload(
        provenance="fallback",
        p50=None,
        p80=None,
        p95=None,
        source=None,
        n=None,
        confidence=None,
        trustworthy_flag=False,
        note=f"No duration available for {cause} on {corridor}. Live engine: {reason}.",
    )


def layer2_section(corridor: str, cause: Optional[str] = None) -> dict:
    junction = _junction_for_corridor(corridor, cause)
    hot = csv("frontend/hotspot_rankings.csv")
    ob = csv("frontend/operational_burden.csv")
    out: dict[str, Any] = {"provenance": "precomputed_lookup", "matched_hotspot": junction}
    if junction and hot is not None:
        m = hot[hot["junction"] == junction]
        if not m.empty:
            r = m.iloc[0]
            gi = max(_f(r.get(c)) or -9e9 for c in ["gi_star_h1", "gi_star_h2", "gi_star_h3", "gi_star_h5"])
            out["gi_significance"] = _f(r.get("gi_star_h1"))
            out["gi_max"] = None if gi == -9e9 else gi
            out["sps"] = _f(r.get("sps"))
            out["nhi"] = _f(r.get("nhi"))
    if junction and ob is not None:
        m = ob[ob["junction"] == junction]
        if not m.empty:
            out["OBI"] = _f(m.iloc[0].get("operational_burden_index"))
    out["note"] = (
        f"Highest-burden junction on {corridor} from the precomputed hotspot ranking."
        if junction else f"No mapped junction for {corridor} in the exports."
    )
    return out


def layer3_section(cause: str, corridor: str) -> dict:
    rs = csv("frontend/risk_scores.csv")
    frag = csv("frontend/corridor_fragility.csv")
    dash = csv("layer3_full_dashboard.csv")
    mp = csv("layer3_manpower_recommendations.csv")
    junction = _junction_for_corridor(corridor, cause)
    out: dict[str, Any] = {
        "provenance": "precomputed_lookup",
        "lookup_junction": junction,
        "lookup_corridor": corridor,
        "lookup_cause": cause,
    }

    # risk tier for this cause x corridor (normalized match)
    if rs is not None:
        sub = rs[_match_col(rs, "event_cause", cause) & _match_col(rs, "corridor", corridor)]
        if not sub.empty:
            scores = pd.to_numeric(sub["survival_risk_score"], errors="coerce").dropna()
            if not scores.empty:
                out["dis"] = float(scores.max())
                out["risk_tier"] = _tier(float(scores.max()))
                out["n_events"] = int(len(sub))

    resources: dict[str, Optional[float]] = {}
    source_row = None

    if dash is not None and junction is not None:
        m = dash[dash["junction"].astype(str) == str(junction)]
        if not m.empty:
            source_row = m.iloc[0]
            resources = _resources_from_row(source_row)

    # Fallback to raw manpower recommendations when dashboard LP allocation is zero.
    if _resources_all_zero(resources) and mp is not None and junction is not None:
        m = mp[mp["junction"].astype(str) == str(junction)]
        if not m.empty:
            source_row = m.iloc[0]
            resources = _resources_from_row(source_row)

    # Cause+corridor junction from risk_scores when corridor mapping is weak.
    if _resources_all_zero(resources) and rs is not None:
        sub = rs[_match_col(rs, "event_cause", cause) & _match_col(rs, "corridor", corridor)]
        if not sub.empty and "junction" in sub.columns:
            junc_counts = sub.groupby("junction").size().sort_values(ascending=False)
            for junc in junc_counts.index.astype(str):
                if mp is not None:
                    m = mp[mp["junction"].astype(str) == junc]
                    if not m.empty:
                        cand = _resources_from_row(m.iloc[0])
                        if not _resources_all_zero(cand):
                            junction = junc
                            resources = cand
                            out["lookup_junction"] = junction
                            break

    out.update(resources)
    if source_row is not None and source_row.get("risk_level") is not None:
        out["dashboard_risk_level"] = str(source_row.get("risk_level"))

    # diversion recommendation (precomputed)
    div = csv("layer3_diversion_recommendations.csv")
    if div is not None and junction is not None and "junction" in div.columns:
        m = div[div["junction"].astype(str) == str(junction)]
        if not m.empty:
            cols = [c for c in div.columns if c != "junction"][:4]
            out["diversion_routes"] = {c: (None if pd.isna(m.iloc[0][c]) else str(m.iloc[0][c])) for c in cols}

    # corridor fragility (Hawkes cascade) - precomputed
    if frag is not None:
        m = frag[_match_col(frag, "corridor", corridor)]
        if not m.empty:
            r = m.iloc[0]
            out["fragility"] = {
                "branching_ratio": _f(r.get("branching_ratio")),
                "current_intensity": _f(r.get("current_intensity")),
                "fragility_log": _f(r.get("fragility_log")),
            }

    out["insufficient_evidence"] = _resources_all_zero(resources)
    if out["insufficient_evidence"]:
        out["note"] = (
            f"No Layer 3 resource blueprint for {cause} on {corridor}"
            + (f" (junction {junction})" if junction else "")
            + ". ODS/LP exports have no positive allocation for this combination."
        )
    else:
        out["note"] = (
            f"Resources from Layer 3 exports keyed by junction {junction} "
            f"(highest ODS manpower on {corridor}"
            + (f", cause-filtered)" if cause else ")")
            + "; LP allocation used when positive, else ODS estimates."
        )
    return out


def layer4_section(cause: str, corridor: str) -> dict:
    pe = csv("frontend/planned_event_recommendations.csv")
    out: dict[str, Any] = {
        "provenance": "precomputed_lookup",
        "lookup_cause": cause,
        "lookup_corridor": corridor,
        "applicable": _norm(cause) in PLANNED_CAUSES,
    }

    if not out["applicable"]:
        out["insufficient_evidence"] = True
        out["note"] = (
            "Layer 4 retrieval applies to planned events only "
            "(procession, protest, public_event, vip_movement). "
            f"'{cause}' is served by Layers 1–3 survival and resource models."
        )
        return out

    if pe is not None:
        m = pe[_match_col(pe, "cause", cause) & _match_col(pe, "corridor", corridor)]
        if m.empty:
            m = pe[_match_col(pe, "cause", cause)]
        if not m.empty:
            m = m.copy()
            m["effective_sample_size"] = pd.to_numeric(m.get("effective_sample_size"), errors="coerce").fillna(0)
            m["confidence"] = pd.to_numeric(m.get("confidence"), errors="coerce").fillna(0)
            r = m.sort_values(["effective_sample_size", "confidence"], ascending=False).iloc[0]
            rec_off = _f(r.get("recommended_officers"))
            rec_bar = _f(r.get("recommended_barricades"))
            rec_tow = _f(r.get("recommended_tow_units"))
            ess = _f(r.get("effective_sample_size"))
            out["confidence_tier"] = str(r.get("confidence_band")) if r.get("confidence_band") is not None else None
            out["confidence"] = _f(r.get("confidence"))
            out["IMS"] = _f(r.get("mean_similarity"))
            out["evidence_weight"] = ess
            out["recommended"] = {
                "officers": rec_off,
                "barricades": rec_bar,
                "tow": rec_tow,
            }
            out["abstain"] = str(r.get("abstain_flag"))
            out["matched_corridor"] = str(r.get("corridor"))
            out["insufficient_evidence"] = (
                (rec_off or 0) == 0 and (rec_bar or 0) == 0 and (rec_tow or 0) == 0 and (ess or 0) == 0
            )
            out["note"] = (
                f"Best precedent for {cause} on {corridor}"
                + (f" (matched row corridor: {r.get('corridor')})" if _norm(str(r.get("corridor"))) != _norm(corridor) else "")
                + "."
            )
            return out

    out["insufficient_evidence"] = True
    out["note"] = f"No retrieved precedent for {cause} on {corridor}; the pipeline would abstain or fall back to Layer 3 priors."
    return out


def layer45_section(cause: str) -> dict:
    sr = csv("layer45_scenario_ready_duration.csv")
    sv = csv("layer45_operational_state_vector_normalized.csv")
    out: dict[str, Any] = {"provenance": "precomputed_lookup"}
    if sv is not None and "event_cause" in sv.columns:
        sub = sv[sv["event_cause"] == cause]
        if not sub.empty:
            out["duration_quantiles"] = {
                "p50": _f(pd.to_numeric(sub.get("safe_duration_p50"), errors="coerce").median()),
                "p80": _f(pd.to_numeric(sub.get("safe_duration_p80"), errors="coerce").median()),
                "p95": _f(pd.to_numeric(sub.get("safe_duration_p95"), errors="coerce").median()),
            }
            out["tail_risk_prob"] = _f(pd.to_numeric(sub.get("tail_risk_prob"), errors="coerce").mean())
            nov = sub.get("novelty_flag")
            dft = sub.get("drift_flag")
            out["novelty_flag"] = bool(nov.astype(str).str.lower().isin(["true", "1"]).any()) if nov is not None else None
            out["drift_flag"] = bool(dft.astype(str).str.lower().isin(["true", "1"]).any()) if dft is not None else None
            out["n_events"] = int(len(sub))
            out["note"] = f"Median guarded quantiles and flags across {len(sub)} fused events for cause '{cause}'."
            return out
    if sr is not None:
        out["duration_quantiles"] = {
            "p50": _f(pd.to_numeric(sr.get("safe_duration_p50"), errors="coerce").median()),
            "p80": _f(pd.to_numeric(sr.get("safe_duration_p80"), errors="coerce").median()),
            "p95": _f(pd.to_numeric(sr.get("safe_duration_p95"), errors="coerce").median()),
        }
        out["note"] = "Cause-level fusion not available; city-level median guarded quantiles shown."
    return out


def layer5_section(cause: str) -> dict:
    df = csv("layer5_frontend_export.csv")
    out: dict[str, Any] = {
        "provenance": "precomputed_lookup",
        "note": "Layer 5 MILP runs in minutes (batch-only). The nearest precomputed allocation for this cause is shown; it was solved offline, not live.",
    }
    if df is not None:
        m = df[df["event_cause"] == cause]
        if m.empty:
            m = df
        r = m.iloc[0]
        out["allocation"] = {
            "officers": _f(r.get("officers_allocated")),
            "barricades": _f(r.get("barricades_allocated")),
            "tow": _f(r.get("tow_trucks_allocated")),
            "qru": _f(r.get("qru_allocated")),
            "service_tier": str(r.get("service_tier")),
        }
        out["cvar_before"] = _f(r.get("baseline_cvar"))
        out["cvar_after"] = _f(r.get("optimized_cvar"))
        out["robustness_score"] = _f(r.get("robustness_score"))
    return out


def layer6_section() -> dict:
    health = csv("layer6_model_health_summary.csv")
    drift = csv("layer6_drift_report.csv")
    out: dict[str, Any] = {"provenance": "precomputed_lookup"}
    if health is not None:
        oh = health["overall_health"].dropna()
        out["relevant_health_status"] = str(oh.iloc[0]) if not oh.empty else None
        if "status" in health.columns:
            vc = health["status"].value_counts().to_dict()
            out["status_counts"] = {str(k): int(v) for k, v in vc.items()}
    if drift is not None and "severity" in drift.columns:
        crit = drift[drift["severity"].astype(str).str.lower() == "critical"]
        if not crit.empty:
            r = crit.iloc[0]
            out["relevant_drift_signal"] = {
                "test": str(r.get("test")), "variable": str(r.get("variable")),
                "severity": str(r.get("severity")),
                "recommendation": str(r.get("recommendation"))[:160] if r.get("recommendation") is not None else None,
            }
    out["note"] = "Monitoring is computed over the feedback log offline; the current health snapshot is shown."
    return out


def layer7_section(hour_local: int) -> dict:
    spill = csv("layer7_spillover_centrality.csv")
    eri = csv("layer7_expected_risk_index.csv")
    topk = csv("layer7_top_k_early_warning.csv")
    out: dict[str, Any] = {"provenance": "precomputed_lookup"}

    # use the requested hour to pick the highest-ERI zone at that time-of-day
    zone = None
    eri_val = None
    if eri is not None and "grid_time_utc" in eri.columns and "ERI" in eri.columns:
        e = eri.copy()
        e["__hr"] = pd.to_datetime(e["grid_time_utc"], errors="coerce", utc=True).dt.hour
        e = e[e["__hr"] == int(hour_local)]
        if not e.empty:
            e["ERI"] = pd.to_numeric(e["ERI"], errors="coerce")
            g = e.groupby("zone")["ERI"].mean().sort_values(ascending=False)
            if not g.empty:
                zone = str(g.index[0])
                eri_val = _f(g.iloc[0])
    if zone is None and topk is not None and "zone" in topk.columns:
        zone = str(topk.iloc[0]["zone"])

    out["zone"] = zone
    out["eri"] = eri_val
    if spill is not None and zone is not None:
        m = spill[spill["zone"] == zone]
        if not m.empty:
            out["spillover_centrality"] = _f(m.iloc[0].get("SSC_centrality"))
            out["half_life_hours"] = _f(m.iloc[0].get("half_life_hours"))
    if topk is not None and zone is not None and "zone" in topk.columns:
        m = topk[topk["zone"] == zone]
        out["early_warning"] = (str(m.iloc[0].get("persistence_class")) if not m.empty else None)
    out["note"] = (
        f"Highest expected-risk zone at hour {hour_local} (precomputed ERI; corridor->zone link is not in the exports, so this is time-of-day driven, city-level)."
    )
    return out


def synthesize(sections: dict, scenario: dict) -> dict:
    l3 = sections["layer3_resources"]
    l7 = sections["layer7_spillover"]
    duration_plan = _resolve_duration_plan(sections)
    parts: list[str] = []

    if duration_plan is not None:
        q = str(duration_plan["quantile"]).upper()
        dur = _fmt_duration_min(float(duration_plan["minutes"]))
        parts.append(f"Plan for a {q} clearance of about {dur}")

    if l3.get("risk_tier"):
        parts.append(f"risk tier {l3['risk_tier']}")

    officers, officer_source = _resolve_officer_count(sections)
    hotspot = sections["layer2_spatial"].get("matched_hotspot")
    if officers is not None and officers > 0:
        loc = f" at {hotspot}" if hotspot else ""
        src_label = {"layer3": "L3", "layer4": "L4 retrieval", "layer5": "L5"}.get(
            officer_source or "", officer_source or ""
        )
        parts.append(f"deploy ~{officers:.0f} officers{loc} ({src_label})")

    if l3.get("diversion_routes"):
        parts.append("activate the recommended diversion")
    if l7.get("zone"):
        parts.append(f"watch {l7['zone']} for spillover")

    headline = "; ".join(parts) + "." if parts else "Insufficient matching data for a synthesised recommendation."
    return {
        "headline": headline,
        "scenario": scenario,
        "duration_plan": duration_plan,
        "officer_plan": (
            {"count": officers, "source": officer_source}
            if officers is not None and officers > 0
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Schemas + endpoints
# ---------------------------------------------------------------------------
class Scenario(BaseModel):
    cause: str
    corridor: str
    hour_local: int = 9
    dow_local: int = 0
    requires_road_closure: bool = False
    priority: str = "High"


@app.get("/health")
def health() -> dict:
    eng = engine()
    return {
        "status": "ok",
        "layer1_live": eng.ok,
        "layer1_reason": eng.reason or "live survival engine ready",
        "outputs_present": OUT.exists(),
    }


@app.get("/api/options")
def options() -> dict:
    dl = csv("frontend/duration_lookup.csv")
    frag = csv("frontend/corridor_fragility.csv")
    spill = csv("layer7_spillover_centrality.csv")
    causes = sorted(dl["event_cause"].dropna().unique().tolist()) if dl is not None else []
    corridors = sorted(dl["corridor"].dropna().unique().tolist()) if dl is not None else []
    if frag is not None:
        corridors = sorted(set(corridors) | set(frag["corridor"].dropna().tolist()))
    zones = sorted(spill["zone"].dropna().tolist()) if spill is not None else []
    return _clean({
        "causes": causes,
        "corridors": corridors,
        "zones": zones,
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "priorities": ["High", "Low", "Unknown"],
        "layer1_live": engine().ok,
        "suggested_examples": [
            {"cause": "public_event", "corridor": "Mysore Road", "label": "public_event × Mysore Road"},
            {"cause": "procession", "corridor": "Mysore Road", "label": "procession × Mysore Road"},
            {"cause": "procession", "corridor": "Bannerghata Road", "label": "procession × Bannerghata Road"},
            {"cause": "public_event", "corridor": "Bannerghata Road", "label": "public_event × Bannerghata Road"},
        ],
    })


@app.post("/api/worked-example")
def worked_example(s: Scenario) -> dict:
    scenario = s.model_dump()
    sections = {
        "input": scenario,
        "layer1_duration": layer1_section(s.cause, s.corridor),
        "layer2_spatial": layer2_section(s.corridor, s.cause),
        "layer3_resources": layer3_section(s.cause, s.corridor),
        "layer4_event": layer4_section(s.cause, s.corridor),
        "layer45_fusion": layer45_section(s.cause),
        "layer5_optimization": layer5_section(s.cause),
        "layer6_learning": layer6_section(),
        "layer7_spillover": layer7_section(s.hour_local),
    }
    sections["recommendation"] = synthesize(sections, scenario)
    sections["provenance"] = {
        k: v.get("provenance")
        for k, v in sections.items()
        if isinstance(v, dict) and "provenance" in v
    }
    return _clean(sections)
