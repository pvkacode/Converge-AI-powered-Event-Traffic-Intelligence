"use client";

import { useRef, useState, useEffect } from "react";
import { motion, useScroll, useTransform, useReducedMotion } from "motion/react";
import { MapPin, Target, TrendUp, Lightning } from "@phosphor-icons/react";

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth <= 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);
  return isMobile;
}

export function HeroScrollPreview() {
  const ref = useRef<HTMLDivElement>(null);
  const reduceMotion = useReducedMotion();
  const isMobile = useIsMobile();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });

  const rotate = useTransform(scrollYProgress, [0, 0.5], [20, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.5], isMobile ? [0.85, 0.95] : [0.9, 1]);
  const translate = useTransform(scrollYProgress, [0, 0.5], [40, -40]);

  return (
    <section className="hero-scroll" ref={ref}>
      <div className="hero-inner hero-scroll-inner">
        <motion.div
          style={reduceMotion ? undefined : { y: translate }}
          className="hero-scroll-title"
        >
          <h2 className="hero-section-title">
            From reactive patrol logs to
            <br />
            <span className="hero-scroll-title-accent">predictive intelligence</span>
          </h2>
        </motion.div>

        <motion.div
          style={
            reduceMotion
              ? undefined
              : { rotateX: rotate, scale, transformPerspective: 1200 }
          }
          className="hero-scroll-card"
        >
          <div className="hero-scroll-card-inner">
            <div className="hero-scroll-window">
              <span className="hero-scroll-dot" />
              <span className="hero-scroll-dot" />
              <span className="hero-scroll-dot" />
              <span className="hero-scroll-window-title">converge / overview</span>
            </div>

            <div className="hero-scroll-kpis">
              <div className="hero-scroll-kpi">
                <MapPin size={18} weight="duotone" />
                <div className="hero-scroll-kpi-num">8,173</div>
                <div className="hero-scroll-kpi-label">real incidents</div>
              </div>
              <div className="hero-scroll-kpi">
                <Target size={18} weight="duotone" />
                <div className="hero-scroll-kpi-num">80 / 294</div>
                <div className="hero-scroll-kpi-label">hotspots ranked</div>
              </div>
              <div className="hero-scroll-kpi">
                <TrendUp size={18} weight="duotone" />
                <div className="hero-scroll-kpi-num">49.81%</div>
                <div className="hero-scroll-kpi-label">CVaR reduction</div>
              </div>
              <div className="hero-scroll-kpi">
                <Lightning size={18} weight="duotone" />
                <div className="hero-scroll-kpi-num">7</div>
                <div className="hero-scroll-kpi-label">critical triggers</div>
              </div>
            </div>

            <div className="hero-scroll-chart">
              {[38, 62, 48, 80, 55, 70, 44, 90, 60, 75, 50, 68].map((h, i) => (
                <span key={i} className="hero-scroll-bar" style={{ height: `${h}%` }} />
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
