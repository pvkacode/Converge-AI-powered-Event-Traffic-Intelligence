// Shared between the anti-flash inline script (app/layout.tsx) and
// ThemeProvider so the localStorage key / data-theme attribute name can't
// drift between the two. No window/localStorage access at module scope -
// themeInitScript() only returns a string; it never executes here.
export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "data-theme";

export function isTheme(v: unknown): v is Theme {
  return v === "light" || v === "dark";
}

// Blocking script injected into <head> (no async/defer) so the correct theme
// is applied to <html> before first paint - no flash of the wrong theme.
// Precedence: stored choice > OS preference (prefers-color-scheme) > light.
export function themeInitScript(): string {
  const key = JSON.stringify(THEME_STORAGE_KEY);
  return `(function(){try{var k=${key};var s=localStorage.getItem(k);var t=(s==="light"||s==="dark")?s:((window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches)?"dark":"light");document.documentElement.setAttribute(k,t);document.documentElement.style.colorScheme=t;}catch(e){}})();`;
}
