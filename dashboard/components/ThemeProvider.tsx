"use client";
import { createContext, useContext, useEffect, useState, useCallback } from "react";
import { THEME_STORAGE_KEY, isTheme, type Theme } from "@/lib/theme";

interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
}
const Ctx = createContext<ThemeCtx>({ theme: "light", toggle: () => {} });

// The anti-flash script in app/layout.tsx already resolves the theme
// (stored choice > OS preference > light) and writes it to <html
// data-theme> before hydration runs, so we just read it back here instead
// of recomputing it. SSR-safe: window is undefined during server render, so
// this falls back to the static "light" default that matches the
// server-rendered <html> attribute, avoiding a hydration mismatch.
function getInitialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const attr = document.documentElement.getAttribute(THEME_STORAGE_KEY);
  return isTheme(attr) ? attr : "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute(THEME_STORAGE_KEY, theme);
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  // Live-follow the OS theme, but only until the user makes an explicit
  // choice via toggle(). Precedence: explicit choice (localStorage) > OS
  // preference > light. toggle() is the only place that writes to
  // localStorage, so once it has, localStorage.getItem below is non-null and
  // this listener stops overriding the user's pick.
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e: MediaQueryListEvent) => {
      let hasExplicitChoice = false;
      try {
        hasExplicitChoice = localStorage.getItem(THEME_STORAGE_KEY) != null;
      } catch {
        // Storage unavailable (e.g. private mode) - treat as no explicit choice.
      }
      if (!hasExplicitChoice) setTheme(e.matches ? "dark" : "light");
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const toggle = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "light" ? "dark" : "light";
      try {
        localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        // Storage unavailable - theme still updates in memory for this session.
      }
      return next;
    });
  }, []);

  return <Ctx.Provider value={{ theme, toggle }}>{children}</Ctx.Provider>;
}

export function useTheme() {
  return useContext(Ctx);
}
