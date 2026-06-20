# Layer 7 — Phase 5: Mathematical Specification

**Governing rule:** every score below is a **deterministic composition of already-published L1–L6 quantities**. No parameter is *learned* from data in Layer 7. All weights are fixed, documented constants, tunable by config but never fit. This keeps Layer 7 auditable and prevents it from becoming a covert predictive layer.

Notation: all `*_z` inputs are the robust z-scores already produced in `layer45_operational_state_vector_normalized.csv`. `σ(x) = 1/(1+e^{-x})` is the logistic squash. `clip(x,a,b)` bounds to `[a,b]`.

---

## 5.1 Operational Risk Score (ORS)  — per active site/event

Combines L4.5 predictive signals, L5 optimization residual risk, and L6 trust into one `[0,100]` operational priority. Inputs are pre-normalized; weights fixed.

$$
\text{raw\_ORS}_r = a_1 z^{\text{tail}}_r + a_2 z^{\text{hi}}_r + a_3 z^{\text{frag}}_r + a_4 z^{\text{obi}}_r + a_5 z^{\text{nov}}_r + a_6 z^{\text{drift}}_r
$$

with fixed $(a_1..a_6) = (1.2,\,1.0,\,1.0,\,0.8,\,0.6,\,0.5)$ (tail-risk and high-impact dominate, drift/novelty are modifiers — same ordering philosophy as L5's site-weight vector).

Map to bounded score and fold in **realized optimization protection** and **model trust**:

$$
\text{ORS}_r = 100 \cdot \sigma(\text{raw\_ORS}_r) \cdot \underbrace{(1 - 0.5\,\rho_r)}_{\text{robustness discount}} \cdot \underbrace{(1 - 0.3\,(1-R_r))}_{\text{duration reliability}}
$$

- $\rho_r$ = L5 `robustness_score` ∈ [0,1] (a well-protected site is *less* operationally urgent → discounted).
- $R_r$ = L4.5 `duration_reliability` ∈ [0,1] (low reliability inflates residual risk).
- Sites absent from L5 (`not_in_layer5`) skip the robustness discount ($\rho_r=0$) and are flagged.

**Robustness diagnostic (added):** report `ORS_components` (the six additive terms) alongside the score so operators see *which* signal drove urgency. Also emit `ors_confidence = min(coverage_flag_score, R_r)` to down-rank scores built on thin upstream coverage.

---

## 5.2 Alert Severity Score (ASS) — per alert

Unifies the **non-uniform** upstream severity vocabularies (`critical/moderate/info`, `critical/warning/none`, `CRITICAL/warning/healthy`) onto one ordinal base, then escalates by corroboration and recency.

**Base severity map** $s_0$:

| Upstream label | $s_0$ |
|---|---|
| critical / CRITICAL | 1.00 |
| moderate / warning | 0.55 |
| info / none / healthy | 0.20 |

**Score (PATCH F-002 — documents the IMPLEMENTED multiplicative form, now bounded).**
The deployed engine uses a multiplicative model, not the earlier additive draft. ASS is
computed as a raw product and then rescaled to $[0,1]$ by its theoretical maximum:

$$
\text{ASS}^{\text{raw}}_a = b(a)\cdot c_a \cdot r_a, \qquad
\text{ASS}_a = \mathrm{clip}\!\left(\frac{\text{ASS}^{\text{raw}}_a}{\text{ASS}_{\max}},\ 0,\ 1\right)
$$

- $b(a)$ = base severity: `critical` 0.90, `moderate` 0.65, `warning` 0.50, `info` 0.25 (unknown → 0.25).
- $c_a = 1 + \beta_c\,n^{\text{src}}_a$ — corroboration factor, $\beta_c = 0.15$, $n^{\text{src}}_a$ = number of distinct source feeds on the same topic.
- $r_a = \exp(-\Delta t_a / h)$ — recency, half-life $h = 24$ h, $\Delta t_a$ from `generated_at` (or file mtime where absent).
- $\text{ASS}_{\max} = \max_b \cdot (1 + \beta_c\,N_{\max}) \cdot 1$ with $N_{\max}=4$ source feeds $= 0.90\times1.60 = 1.44$ — guarantees $\text{ASS}_a \in [0,1]$ (monotone rescale; ordering preserved).

**Priority (PATCH F-003 — quantile-based).** Priority is assigned by the percentile of the
bounded ASS over the deduplicated feed, not a fixed absolute cut (which let every
corroborated critical reach P1): `P1` = top 15% ($\ge 0.85$ pct), `P2` = 55–85%,
`P3` = 25–55%, `P4` = bottom 25%.

**Discrete tier** for the UI maps from the P1–P4 bands above.

**Robustness diagnostic:** `ass_dedup_group` (hash of layer+variable+event) so the feed is deduplicated; keep the max-ASS representative and a `merged_count`.

---

## 5.3 Override Impact Score (OIS) — per human override

Quantifies how far an operator's manual change moves the plan away from the optimizer's solution, in risk-equivalent units, **without re-solving the MILP**.

For an override that changes allocation vector $x_r \to x'_r$ at site $r$:

$$
\Delta\text{eff}_r = E_r(x'_r) - E_r(x_r), \qquad E_r(u)=1-\exp\!\big(-(\gamma_p p_r+\gamma_b b_r+\gamma_t t_r+\gamma_q q_r)\big)
$$

reusing L5's **published** effectiveness coefficients $(\gamma_p,\gamma_b,\gamma_t,\gamma_q)=(0.18,0.10,0.25,0.30)$ (read-only; identical to L5 — not refit).

Projected per-site delay change (using L5 site weight $w_r$ and the site's expected scenario duration $\mathbb{E}[T_r]$, both from L5 outputs):

$$
\Delta D_r = -\,w_r\,\mathbb{E}[T_r]\,\Delta\text{eff}_r
$$

$$
\boxed{\ \text{OIS}_r = \underbrace{\Delta D_r}_{\text{expected delay change}} + \lambda_{cc}\,\Delta\text{CCV}_r + \lambda_{b}\,\Delta\text{Budget}_r\ }
$$

- $\Delta\text{CCV}_r$ = change in chance-constraint margin (negative if the override breaks the site's tier minimum-service or budget cap) — flagged as **override violation** if it pushes below L5's tier minimums.
- $\Delta\text{Budget}_r$ = signed budget consumption vs caps (120 officers / 100 barricades / 15 tow / 10 qru).
- Fixed penalties $(\lambda_{cc},\lambda_b)=(2.0,\,0.15)$ mirror L5's objective weights for commensurability.

A **positive OIS** means the override is expected to *increase* delay/risk vs the optimizer (operator override is costly); **negative** means the operator improved on the plan under their private information. OIS is descriptive accounting, never a gate.

---

## 5.4 Sensor Fusion (future-only, defined now for the seam)

Today the only "sensors" are the L4.5 JOSV signals. The fusion operator is a **precision-weighted (inverse-variance) combination** so real sensors can be added later without changing the formula:

$$
\hat{s}_r = \frac{\sum_k \omega_{k,r}\, s_{k,r}}{\sum_k \omega_{k,r}}, \qquad \omega_{k,r} = \frac{c_{k,r}}{\sigma_{k}^2}
$$

where $s_{k,r}$ is sensor/source $k$'s estimate for signal at site $r$, $\sigma_k^2$ its (declared) noise variance, and $c_{k,r}\in\{0,1\}$ a coverage/freshness gate. With a single source (JOSV) this collapses to identity — correct degenerate behavior. **No Kalman/learned fusion** (that would be a new model).

---

## 5.5 Digital Twin dynamics (research-grade; re-evaluation, not re-solving)

Given operator-perturbed inputs (e.g. "what if officer budget −20%", "what if tail_risk at site r +0.1"), the twin **re-evaluates L5's published closed-form objective** on the existing scenario matrix, holding the integer allocation fixed (or applying a deterministic greedy re-fill identical to L5's documented fallback). It never re-runs HiGHS.

Per-scenario delay (L5 formula, verbatim):

$$
D_s(x) = \sum_r w_r\,T_{r,s}\,(1-e_r), \qquad
\text{CVaR}_\alpha = z + \frac{1}{(1-\alpha)S}\sum_s \xi_s,\ \ \xi_s=\max(0, D_s - z)
$$

Twin reports $\Delta$CVaR and $\Delta$expected-delay between baseline and perturbed inputs. **Uncertainty inflation** for low-reliability sites reuses L5's published rule $\sigma_r^{\text{adj}}=\sigma_r(1+\kappa(1-R_r))$, $\kappa=0.5$. This is simulation over fixed math, not a new predictive layer.

---

## 5.6 What Layer 7 math may and may not do
- **May:** weighted sums, normalization, logistic squashing, recency decay, inverse-variance fusion, deterministic re-evaluation of published L5 formulas, rule-based tiering/dedup.
- **May add:** robustness/coverage diagnostics, confidence gates, provenance fields.
- **May NOT:** fit/learn any parameter, retrain any model, alter L1–L6 formulas, or re-solve the MILP. All constants above are fixed defaults living in a Layer-7 config module, changeable only by humans.
