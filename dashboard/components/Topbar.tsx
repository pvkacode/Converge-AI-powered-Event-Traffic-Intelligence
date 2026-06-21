"use client";
import { usePathname } from "next/navigation";
import { Sun, Moon } from "@phosphor-icons/react";
import { useTheme } from "./ThemeProvider";
import { navTitle, NAV } from "@/lib/nav";

export function Topbar() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();
  const title = navTitle(pathname);
  const crumb = NAV.find((n) => n.href === pathname)?.idx ?? "";

  return (
    <header className="topbar">
      <div className="row gap-3">
        <span className="topbar-crumb">{crumb || "-"}</span>
        <span className="topbar-title">{title}</span>
      </div>
      <div className="topbar-actions">
        <span className="dim mono" style={{ fontSize: 11 }}>
          read-only frontend
        </span>
        <button
          className="iconbtn"
          onClick={toggle}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
        >
          {theme === "light" ? <Moon size={17} weight="bold" /> : <Sun size={17} weight="bold" />}
        </button>
      </div>
    </header>
  );
}
