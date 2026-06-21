# Converge / ASTraM dashboard

A read-only operations frontend over the Bengaluru traffic disruption ML pipeline. It reads
the real CSV exports from the sibling `outputs/` directory at runtime and renders them as an
interactive, themeable dashboard. It does not modify, regenerate, or depend on any pipeline
code under `src/` or `data/`.

## Run

```bash
cd dashboard
npm install
npm run dev      # http://localhost:3000
```

For a production build:

```bash
npm run build && npm run start
```

The app resolves `outputs/` relative to the repo root automatically (it looks for
`../outputs/frontend`). Nothing is bundled or copied; files are read live via the Node `fs`
API in `lib/csv.ts`, with mtime-aware caching.

## What it shows

- **Overview** — pipeline flow diagram and live KPIs (hotspots, median duration, model
  health, active alerts, top spillover zone).
- **Layer 1 – 7** — one page per model layer, each with a plain-language header, KPIs, real
  charts (recharts), and sortable / filterable / paginated tables driven by `/api/dataset`.
- **Worked example** — a fixed vehicle-breakdown-on-Mysore-Road scenario traced through
  Layers 1 → 2 → 3 as a clickable step sequence.

## Design

- Warm-yellow surface palette (cornsilk / lemon chiffon / vanilla / jasmine), warm-charcoal
  ink, a single deep-teal accent, and muted palette-aware status colours. All defined once as
  CSS variables in `app/globals.css`.
- Light and dark themes, toggled app-wide from the top bar. The choice is in-memory React
  state only (no `localStorage`).

## Honesty notes (surfaced in the UI, not silently fixed)

- `duration_lookup.csv` contains right-censored extreme durations for some cause/corridor
  cells; shown verbatim with a caveat on Layer 1.
- `layer5_pareto_front.csv` ships with only a header row, so the Pareto chart shows an honest
  empty state and the page renders the real CVaR-vs-alpha frontier instead.
- The Layer 3 risk tier is a derived presentation label (binned on the empirical
  `survival_risk_score` distribution), clearly marked as derived.
