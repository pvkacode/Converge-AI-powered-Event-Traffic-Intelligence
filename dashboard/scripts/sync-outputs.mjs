#!/usr/bin/env node
/**
 * Copy repo-root outputs/ into dashboard/outputs for Vercel deploys.
 * Vercel scopes serverless bundles to the Root Directory; sibling ../outputs
 * is not included unless copied here during build.
 *
 * Skipped locally unless VERCEL=1 or SYNC_OUTPUTS=1.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dashboardRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoOutputs = path.join(dashboardRoot, "..", "outputs");
const localOutputs = path.join(dashboardRoot, "outputs");

const shouldSync =
  process.env.VERCEL === "1" ||
  process.env.SYNC_OUTPUTS === "1" ||
  process.argv.includes("--force");

if (!shouldSync) {
  console.log("[sync-outputs] skipped (set VERCEL=1 or SYNC_OUTPUTS=1 to copy)");
  process.exit(0);
}

if (!fs.existsSync(repoOutputs)) {
  console.warn(`[sync-outputs] source not found: ${repoOutputs}`);
  process.exit(0);
}

if (fs.existsSync(localOutputs)) {
  fs.rmSync(localOutputs, { recursive: true, force: true });
}

fs.cpSync(repoOutputs, localOutputs, { recursive: true });
const n = fs.readdirSync(localOutputs).length;
console.log(`[sync-outputs] copied ${repoOutputs} → ${localOutputs} (${n} top-level entries)`);
