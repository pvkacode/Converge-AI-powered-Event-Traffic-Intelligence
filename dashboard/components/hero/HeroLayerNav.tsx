"use client";

import Link from "next/link";
import { Sun, Moon } from "@phosphor-icons/react";
import { useTheme } from "@/components/ThemeProvider";
import { ApiStatusDot } from "@/components/ApiStatusDot";
import { ALL_LAYERS, LAYER_COLORS } from "./constants";

export function HeroLayerNav() {
  const { theme, toggle } = useTheme();

  return (
    <nav className="hero-top-nav" aria-label="Jump to pipeline layer">
      <div className="hero-top-nav-inner">
        <Link href="/overview" className="hero-top-brand" title="Open operations dashboard">
          <span className="hero-top-brand-mark">CV</span>
          <span className="hero-top-brand-text">Converge</span>
        </Link>

        <div className="hero-top-links" role="list">
          {ALL_LAYERS.map((layer) => {
            const color = LAYER_COLORS[layer.id as keyof typeof LAYER_COLORS];
            return (
              <Link
                key={layer.id}
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
            );
          })}
        </div>

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
