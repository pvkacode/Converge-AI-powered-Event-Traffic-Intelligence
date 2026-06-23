"use client";

import { useEffect, useState } from "react";
import { BACKEND_SLOW_MS } from "@/lib/backend-notice";

/** True once `active` has stayed true for longer than `thresholdMs`. */
export function useSlowLoading(active: boolean, thresholdMs = BACKEND_SLOW_MS): boolean {
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    if (!active) {
      setSlow(false);
      return;
    }
    const timer = window.setTimeout(() => setSlow(true), thresholdMs);
    return () => window.clearTimeout(timer);
  }, [active, thresholdMs]);

  return slow;
}
