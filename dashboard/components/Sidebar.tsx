"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV } from "@/lib/nav";

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <div className="brand-dot">CV</div>
          <div>
            <div className="brand-name">Converge</div>
            <div className="brand-sub">ASTraM · Bengaluru</div>
          </div>
        </div>
      </div>
      <nav className="nav" aria-label="Primary">
        <div className="nav-group-label">Pipeline</div>
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item${active ? " active" : ""}`}
              aria-current={active ? "page" : undefined}
            >
              <span className="nav-idx">{item.idx}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="sidebar-foot">read-only · outputs/</div>
    </aside>
  );
}
