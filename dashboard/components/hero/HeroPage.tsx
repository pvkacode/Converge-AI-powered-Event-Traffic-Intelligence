"use client";

import Link from "next/link";
import {
  MapPin,
  Target,
  Users,
  TrendUp,
  CaretDown,
  Clock,
  MapTrifold,
  Wrench,
  MagnifyingGlass,
  Graph,
  Scales,
  Lightning,
  Brain,
  Database,
  Stack,
  XCircle,
  Pulse,
  GithubLogo,
  Warning,
  HandPointing,
} from "@phosphor-icons/react";
import { MiniDemo } from "./MiniDemo";
import { HeroLayerNav } from "./HeroLayerNav";
import { PIPELINE_LAYERS, LAYER_COLORS } from "./constants";
import { useCountUp, useInView, scrollToId, useMounted } from "./hooks";
import "./hero.css";

const LAYER_ICONS: Record<string, React.ReactNode> = {
  L1: <Clock size={28} weight="duotone" />,
  L2: <MapTrifold size={28} weight="duotone" />,
  L3: <Wrench size={28} weight="duotone" />,
  L4: <MagnifyingGlass size={28} weight="duotone" />,
  "L4.5": <Graph size={28} weight="duotone" />,
  L5: <Scales size={28} weight="duotone" />,
  L7: <Lightning size={28} weight="duotone" />,
};

function HeroStat({
  icon,
  end,
  label,
  isDecimal,
  enabled,
}: {
  icon: React.ReactNode;
  end: number;
  label: string;
  isDecimal?: boolean;
  enabled: boolean;
}) {
  const val = useCountUp(end, 1500, enabled, isDecimal, 2);
  return (
    <div className="hero-stat">
      <div className="hero-stat-icon">{icon}</div>
      <div className="hero-stat-num">
        {val}
        {isDecimal ? "%" : ""}
      </div>
      <div className="hero-stat-label">{label}</div>
    </div>
  );
}

function ResultMetric({
  icon,
  display,
  suffix,
  label,
  sub,
  enabled,
  children,
}: {
  icon: React.ReactNode;
  display: React.ReactNode;
  suffix?: string;
  label: string;
  sub?: string;
  enabled: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div className="hero-result-item">
      <div style={{ color: "var(--hero-accent)", marginBottom: 8 }}>{icon}</div>
      <div className="hero-result-num">
        {display}
        {suffix}
      </div>
      {children}
      <div className="hero-result-label">{label}</div>
      {sub ? <div className="hero-result-sub">{sub}</div> : null}
    </div>
  );
}

function CountResult({
  end,
  suffix,
  label,
  sub,
  enabled,
  icon,
  decimals,
}: {
  end: number;
  suffix?: string;
  label: string;
  sub?: string;
  enabled: boolean;
  icon: React.ReactNode;
  decimals?: number;
}) {
  const isDec = decimals != null;
  const val = useCountUp(end, 1200, enabled, isDec, decimals ?? 0);
  return (
    <ResultMetric
      icon={icon}
      display={val}
      suffix={suffix}
      label={label}
      sub={sub}
      enabled={enabled}
    />
  );
}

function CriticalTriggersMetric({ enabled }: { enabled: boolean }) {
  const val = useCountUp(7, 1200, enabled);
  return (
    <div className="hero-result-item">
      <div style={{ color: "var(--hero-accent)", marginBottom: 8 }}>
        <Warning size={22} weight="duotone" />
      </div>
      <div className="hero-result-num">{val}</div>
      {enabled ? (
        <span className="hero-pill hero-pill-critical" style={{ marginTop: 6, display: "inline-flex" }}>
          CRITICAL
        </span>
      ) : null}
      <div className="hero-result-label">critical retrain triggers</div>
    </div>
  );
}

function PValueMetric({ enabled }: { enabled: boolean }) {
  const text = useCountUp(0, 1, false);
  const { ref, inView } = useInView(0.15);
  const show = enabled && inView;
  return (
    <div className="hero-result-item" ref={ref as React.RefObject<HTMLDivElement>}>
      <div style={{ color: "var(--hero-accent)", marginBottom: 8 }}>
        <Lightning size={22} weight="duotone" />
      </div>
      <div className="hero-result-num" style={{ fontSize: 26 }}>
        {show ? (
          <>
            2.5 × 10<sup style={{ fontSize: 14 }}>−91</sup>
          </>
        ) : (
          text
        )}
      </div>
      <div className="hero-result-label">spillover p-value</div>
    </div>
  );
}

export function HeroPage() {
  const mounted = useMounted(50);
  const titleOn = useMounted(0);
  const subOn = useMounted(200);
  const ctaOn = useMounted(400);
  const statsOn = useMounted(600);

  const pipelineView = useInView(0.15);
  const diffView = useInView(0.15);
  const resultsView = useInView(0.15);

  return (
    <div className="hero">
      <HeroLayerNav />
      {/* ── Section 1: Hero banner ── */}
      <section className="hero-banner">
        <div className="hero-banner-grid">
          <div>
            <p className="hero-kicker">Bengaluru Traffic Police · ASTraM · Gridlock 2.0</p>
            <h1 className={`hero-title${titleOn ? " is-visible" : ""}`}>CONVERGE</h1>
            <p className={`hero-subtitle${subOn ? " is-visible" : ""}`}>
              From reactive patrol logs to a predictive, self-improving disruption-intelligence
              system for Bengaluru traffic.
            </p>
            <div className={`hero-ctas${ctaOn ? " is-visible" : ""}`}>
              <button type="button" className="hero-btn hero-btn-primary" onClick={() => scrollToId("hero-system")}>
                Explore the Pipeline
              </button>
              <button type="button" className="hero-btn hero-btn-secondary" onClick={() => scrollToId("hero-demo")}>
                Run Live Inference
              </button>
            </div>
            <div className={`hero-stats${statsOn ? " is-visible" : ""}`}>
              <HeroStat icon={<MapPin size={18} weight="duotone" />} end={8173} label="real incidents" enabled={statsOn} />
              <HeroStat icon={<Target size={18} weight="duotone" />} end={80} label="hotspots found" enabled={statsOn} />
              <HeroStat icon={<Users size={18} weight="duotone" />} end={294} label="junctions mapped" enabled={statsOn} />
              <HeroStat icon={<TrendUp size={18} weight="duotone" />} end={49.81} label="CVaR reduction" isDecimal enabled={statsOn} />
            </div>
          </div>

          <div className="hero-visual">
            <svg className="hero-radar" viewBox="0 0 400 400" aria-hidden>
              <path
                className="hero-arc"
                d="M 380 380 A 400 400 0 0 0 80 120"
                fill="none"
                stroke="var(--hero-gold)"
                strokeWidth="2"
                opacity="0.2"
              />
              <path
                className="hero-arc"
                d="M 380 380 A 300 300 0 0 0 120 160"
                fill="none"
                stroke="var(--hero-gold)"
                strokeWidth="2"
                opacity="0.35"
              />
              <path
                className="hero-arc"
                d="M 380 380 A 200 200 0 0 0 200 200"
                fill="none"
                stroke="var(--hero-gold)"
                strokeWidth="2"
                opacity="0.55"
              />
            </svg>

            <div
              className={`hero-float-card${mounted ? " is-visible" : ""}`}
              style={{ top: 24, left: 0 }}
            >
              <span className="hero-pill hero-pill-critical">CRITICAL</span>
              <div style={{ fontSize: 13, fontWeight: 600, marginTop: 6 }}>Model Health</div>
              <div className="dim" style={{ fontSize: 11 }}>3 of 20 checks</div>
            </div>

            <div
              className={`hero-float-card${mounted ? " is-visible" : ""}`}
              style={{ top: "38%", left: "18%" }}
            >
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--hero-accent)" }}>
                Central Zone 2
              </div>
              <div className="dim" style={{ fontSize: 11 }}>Top Spillover Zone</div>
            </div>

            <div
              className={`hero-float-card${mounted ? " is-visible" : ""}`}
              style={{ bottom: 32, right: 0 }}
            >
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 700, color: "var(--hero-gold)" }}>
                p ≈ 10⁻⁹¹
              </div>
              <div className="dim" style={{ fontSize: 11 }}>spillover confirmed</div>
            </div>
          </div>
        </div>

        <div className="hero-scroll-hint" aria-hidden>
          <CaretDown size={24} />
        </div>
      </section>

      {/* ── Section 2: Pipeline ── */}
      <section className="hero-pipeline" id="hero-system" ref={pipelineView.ref as React.RefObject<HTMLElement>}>
        <div className="hero-inner">
          <p className="hero-eyebrow">The System</p>
          <h2 className="hero-section-title">Seven layers. One decision.</h2>
          <p className="hero-section-sub">
            Every layer feeds the next — from raw incident data to a CVaR-bounded operational plan.
          </p>
          <p className="hero-pipeline-hint">
            <HandPointing size={16} weight="duotone" aria-hidden />
            Click any layer card to open its full dashboard — outputs, tables, maps, and methodology.
          </p>

          <div className="hero-pipeline-row">
            {PIPELINE_LAYERS.map((layer, i) => {
              const color = LAYER_COLORS[layer.id as keyof typeof LAYER_COLORS];
              const visible = pipelineView.inView;
              return (
                <div key={layer.id} style={{ display: "contents" }}>
                  <Link
                    href={layer.href}
                    className={`hero-layer-card${visible ? " is-visible" : ""}`}
                    style={{
                      background: `color-mix(in srgb, ${color} 15%, transparent)`,
                      border: `1px solid color-mix(in srgb, ${color} 60%, transparent)`,
                      transitionDelay: visible ? `${i * 100}ms` : undefined,
                    }}
                  >
                    <span className="hero-layer-label" style={{ color }}>
                      {layer.id}
                    </span>
                    <span className="hero-layer-icon" style={{ color }}>
                      {LAYER_ICONS[layer.id]}
                    </span>
                    <span className="hero-layer-name">{layer.title}</span>
                    <span className="hero-layer-tech">{layer.techniques}</span>
                    <span className="hero-layer-tooltip">{layer.metric}</span>
                  </Link>
                  {i < PIPELINE_LAYERS.length - 1 && (
                    <div className="hero-pipe-arrow">
                      <div className="hero-pipe-arrow-line">
                        <span className={`hero-pipe-dot${visible ? " is-active" : ""}`} style={{ animationDelay: `${i * 100 + 400}ms` }} />
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div className="hero-l6-arc">
            <svg viewBox="0 0 800 64" preserveAspectRatio="none" aria-hidden>
              <path
                id="l6-arc-path"
                d="M 20 48 Q 400 4 780 48"
                fill="none"
                stroke="var(--hero-accent)"
                strokeWidth="2"
                strokeDasharray="8 6"
                opacity="0.6"
              />
              <circle r="5" fill="var(--hero-accent)" className="hero-l6-dot">
                <animateMotion dur="3s" repeatCount="indefinite" path="M 780 48 Q 400 4 20 48" />
              </circle>
            </svg>
            <div className="hero-l6-label">
              <Brain size={18} weight="duotone" />
              L6 — Adaptive Learning closes the loop
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 3: Mini demo ── */}
      <MiniDemo />

      {/* ── Section 4: Differentiators ── */}
      <section className="hero-diff" ref={diffView.ref as React.RefObject<HTMLElement>}>
        <div className="hero-inner">
          <p className="hero-eyebrow">What makes this different</p>
          <h2 className="hero-section-title">Built for operators, documented for skeptics</h2>

          <div className="hero-diff-grid">
            <div
              className={`hero-diff-card${diffView.inView ? " is-visible" : ""}`}
              style={{ transitionDelay: diffView.inView ? "0ms" : undefined }}
            >
              <div className="hero-icon-circle" style={{ background: "var(--hero-accent-soft)", color: "var(--hero-accent)" }}>
                <Database size={22} weight="duotone" />
              </div>
              <h3 className="hero-diff-title">Real data, cleaned honestly</h3>
              <p className="hero-diff-body">
                8,173 ASTraM incidents from Bengaluru Traffic Police. Right-censoring modeled via
                Kaplan-Meier, not hidden. Every row weighted by trust_score — none deleted.
              </p>
              <div className="hero-diff-stat" style={{ color: "var(--hero-accent)" }}>
                4,500+ censored rows handled correctly
              </div>
            </div>

            <div
              className={`hero-diff-card${diffView.inView ? " is-visible" : ""}`}
              style={{ transitionDelay: diffView.inView ? "150ms" : undefined }}
            >
              <div className="hero-icon-circle" style={{ background: "color-mix(in srgb, var(--hero-gold) 20%, transparent)", color: "var(--hero-gold)" }}>
                <Stack size={22} weight="duotone" />
              </div>
              <h3 className="hero-diff-title">Seven layers, one decision</h3>
              <p className="hero-diff-body">
                Survival analysis feeds spatial intelligence feeds retrieval feeds stochastic
                optimization — with documented handoffs and leak-free feature construction at every
                boundary.
              </p>
              <div className="hero-diff-stat" style={{ color: "var(--hero-gold)" }}>
                350+ MILP decision variables
              </div>
            </div>

            <div
              className={`hero-diff-card is-highlight${diffView.inView ? " is-visible" : ""}`}
              style={{ transitionDelay: diffView.inView ? "300ms" : undefined }}
            >
              <div className="hero-icon-circle" style={{ background: "var(--critical-bg)", color: "var(--critical)" }}>
                <XCircle size={22} weight="duotone" />
              </div>
              <h3 className="hero-diff-title">We report what didn&apos;t work</h3>
              <p className="hero-diff-body">
                Stacked ensemble removed — didn&apos;t beat RSF. Fallback blending removed — made RMSE
                worse. Gi* z-cutoff bug caught and fixed. Negative results documented, not hidden.
              </p>
              <div className="hero-diff-stat" style={{ color: "var(--critical)" }}>
                5 honest negative results, all in the repo
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Section 5: Results strip ── */}
      <section className="hero-results" ref={resultsView.ref as React.RefObject<HTMLElement>}>
        <div className="hero-results-row">
          <CountResult
            icon={<MapPin size={22} weight="duotone" />}
            end={80}
            suffix=" / 294"
            label="significant hotspots"
            enabled={resultsView.inView}
          />
          <CountResult
            icon={<Pulse size={22} weight="duotone" />}
            end={0.7}
            label="RSF C-index"
            sub="vs 0.50 random"
            enabled={resultsView.inView}
            decimals={2}
          />
          <CountResult
            icon={<Clock size={22} weight="duotone" />}
            end={40.9}
            suffix=" min"
            label="planned-event MAE"
            enabled={resultsView.inView}
            decimals={1}
          />
          <CountResult
            icon={<Target size={22} weight="duotone" />}
            end={57.6}
            suffix="%"
            label="within 20 min of actual"
            enabled={resultsView.inView}
            decimals={1}
          />
          <PValueMetric enabled={resultsView.inView} />
          <CriticalTriggersMetric enabled={resultsView.inView} />
        </div>
      </section>

      {/* ── Section 6: Footer ── */}
      <footer className="hero-footer">
        <div className="hero-footer-team">
          <strong>Team Converge</strong>
          <span className="dim"> · pvkacode · Shrija Tewari · Saloni Singh · adityagayake</span>
        </div>
        <div className="hero-footer-links">
          <a
            className="hero-github-btn"
            href="https://github.com/pvkacode/Converge-AI-powered-Event-Traffic-Intelligence"
            target="_blank"
            rel="noopener noreferrer"
          >
            <GithubLogo size={18} weight="duotone" />
            View Repository
          </a>
          <span className="hero-footer-meta">
            Gridlock 2.0 · Event-Driven Congestion Track · Bengaluru Traffic Police
          </span>
        </div>
      </footer>
    </div>
  );
}
