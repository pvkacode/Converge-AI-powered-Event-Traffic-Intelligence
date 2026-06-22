"use client";

import Link from "next/link";
import { ALL_LAYERS, LAYER_COLORS } from "./constants";

export function HeroLayerNav() {
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

        <Link href="/worked-example" className="hero-top-wx" title="Worked Example — full inference trace">
          WX
        </Link>
      </div>
    </nav>
  );
}
