"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { Badge } from "./ui";

type ApiStatus = "checking" | "online" | "offline";

const POLL_MS = 25_000;
const TIMEOUT_MS = 4_000;

// Purely informational — never blocks or throws. All failure modes (network
// error, non-2xx, timeout) resolve to "offline" so a dead Render free-tier
// instance never crashes the page it's mounted on.
export function ApiStatusDot() {
  const [status, setStatus] = useState<ApiStatus>("checking");
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
      try {
        const res = await fetch(`${API_BASE}/health`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!cancelled) setStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      } finally {
        clearTimeout(timeout);
      }
    }

    check();
    const interval = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
      controllerRef.current?.abort();
    };
  }, []);

  const variant = status === "online" ? "ok" : status === "checking" ? "muted" : "warning";
  const label = status === "online" ? "API online" : status === "checking" ? "Checking…" : "Backend waking up…";
  const aria =
    status === "online"
      ? "API online"
      : status === "checking"
      ? "Checking API status"
      : "API offline — backend may be waking up";

  return (
    <span title={aria} aria-label={aria}>
      <Badge variant={variant} dot>
        {label}
      </Badge>
    </span>
  );
}
