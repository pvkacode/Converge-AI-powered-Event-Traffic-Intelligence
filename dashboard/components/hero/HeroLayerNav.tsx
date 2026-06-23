"use client";

import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { Sun, Moon } from "@phosphor-icons/react";
import { useTheme } from "@/components/ThemeProvider";
import { ApiStatusDot } from "@/components/ApiStatusDot";
import { ALL_LAYERS, LAYER_COLORS } from "./constants";

const navContainer = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12, delayChildren: 0.3 } },
};

const navChip = {
  hidden: { opacity: 0, x: -20 },
  visible: { opacity: 1, x: 0, transition: { duration: 0.6, ease: [0.25, 0.4, 0.25, 1] as const } },
};

const navChipInstant = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0 } },
};

export function HeroLayerNav() {
  const { theme, toggle } = useTheme();
  const reduceMotion = useReducedMotion();
  const chip = reduceMotion ? navChipInstant : navChip;

  return (
    <nav className="hero-top-nav" aria-label="Jump to pipeline layer">
      <div className="hero-top-nav-inner">
        <motion.div
          className="hero-top-links"
          role="list"
          variants={navContainer}
          initial="hidden"
          animate="visible"
        >
          {ALL_LAYERS.map((layer) => {
            const color = LAYER_COLORS[layer.id as keyof typeof LAYER_COLORS];
            return (
              <motion.div key={layer.id} variants={chip}>
                <Link
                  href={layer.href}
                  className="hero-top-link"
                  role="listitem"
                  title={layer.title}
                  style={
                    {
                      "--layer-color": color,
                    } as React.CSSProperties
                  }
                >
                  <span className="hero-top-link-id">{layer.id}</span>
                  <span className="hero-top-link-name">{layer.title}</span>
                </Link>
              </motion.div>
            );
          })}
        </motion.div>

        <div className="hero-top-actions">
          <ApiStatusDot />
          <Link href="/worked-example" className="hero-top-wx" title="Worked Example — full inference trace">
            WX
          </Link>
          <button
            type="button"
            className="hero-top-theme iconbtn"
            onClick={toggle}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? <Moon size={17} weight="bold" /> : <Sun size={17} weight="bold" />}
          </button>
        </div>
      </div>
    </nav>
  );
}
