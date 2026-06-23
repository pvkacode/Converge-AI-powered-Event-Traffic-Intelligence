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
- Light and dark themes, toggled app-wide from the top bar. The choice persists across
  reloads via `localStorage` (falling back to the OS `prefers-color-scheme` on first visit)
  and is applied before first paint, so there's no flash of the wrong theme. See
  [Recent improvements](#recent-improvements).

## Honesty notes (surfaced in the UI, not silently fixed)

- `duration_lookup.csv` contains right-censored extreme durations for some cause/corridor
  cells; shown verbatim with a caveat on Layer 1.
- `layer5_pareto_front.csv` ships with only a header row, so the Pareto chart shows an honest
  empty state and the page renders the real CVaR-vs-alpha frontier instead.
- The Layer 3 risk tier is a derived presentation label (binned on the empirical
  `survival_risk_score` distribution), clearly marked as derived.

## Recent improvements

All display-only; nothing under `src/` or `data/` was touched, and no CSV schema changed.

- **Layer 1 — cause-level fallback badge.** `lookup_expected_duration()` in
  `src/layer1_survival.py` falls back to a cause-only prior when a (cause, corridor) pair
  doesn't clear `MIN_GROUP_SIZE`, but nothing in the UI said so. The duration explorer
  (`components/DurationExplorer.tsx`) now derives this on the frontend — a pair absent from
  `duration_lookup.csv` but present in `layer1_survival_fallback.csv` is flagged with a
  "Cause-level estimate — sparse corridor history" badge (`components/
  DurationConfidenceBadge.tsx`, `lib/durationFallback.ts`). Thin-but-resolved corridor
  estimates (`n < 30`) get a distinct "Low-confidence estimate" badge instead. True
  corridor-specific rows render exactly as before.
- **Layer 6 — retrain triggers surfaced.** `outputs/layer6_retrain_triggers.csv` was produced
  by the pipeline but never rendered, so findings like "8 critical retrain triggers" were
  invisible in the UI. Layer 6 now has a sortable, filterable trigger table
  (`components/SeverityEventTable.tsx`, `components/Layer6TriggerPanel.tsx`,
  `lib/severity.ts`) with severity filter chips (All/Critical/Moderate/Info), free-text
  search, click-to-sort columns (default: critical first, then by score), and expandable
  rows showing the full trigger detail (signal, test, recommended action). A second tab
  shows `layer6_active_alerts.csv` priority-ranked the same way, and hides itself entirely
  if that file is ever absent.
- **Theme persistence, no flash.** The theme choice now survives reloads. A small
  dependency-free script (`lib/theme.ts`, inlined into `<head>` in `app/layout.tsx`) runs
  before first paint and resolves the theme with precedence **explicit choice (localStorage)
  > OS preference (`prefers-color-scheme`) > light**. `ThemeProvider` reads that
  already-resolved value back on mount instead of recomputing it, and live-follows OS theme
  changes only until the user explicitly toggles — after that, their choice sticks regardless
  of OS changes. All storage access is wrapped in `try/catch` and guarded with
  `typeof window !== "undefined"`, so it degrades to the original in-memory behaviour in
  private/disabled-storage browsers. The public `ThemeProvider` / `useTheme()` API is
  unchanged.
