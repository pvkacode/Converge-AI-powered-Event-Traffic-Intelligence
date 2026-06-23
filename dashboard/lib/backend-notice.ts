import { API_BASE } from "./api";

/** Show “server waking up” copy after this many ms of waiting. */
export const BACKEND_SLOW_MS = 4000;

export function isHostedBackend(): boolean {
  try {
    const host = new URL(API_BASE).hostname.toLowerCase();
    return !["localhost", "127.0.0.1", "::1"].includes(host);
  } catch {
    return false;
  }
}

/** Health URL — opening it helps wake a sleeping Render instance. */
export function getBackendWakeUrl(): string {
  try {
    return `${new URL(API_BASE).origin}/health`;
  } catch {
    return `${API_BASE.replace(/\/$/, "")}/health`;
  }
}

export function getBackendHostLabel(): string {
  try {
    return new URL(API_BASE).host;
  } catch {
    return API_BASE;
  }
}
