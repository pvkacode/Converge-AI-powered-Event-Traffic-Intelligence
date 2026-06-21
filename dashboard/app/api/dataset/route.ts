// Generic dataset API: server-side pagination, sorting, and free-text filtering
// over the real CSV outputs. The `key` is validated against the dataset
// whitelist, so no arbitrary file path can be read.
import { NextRequest, NextResponse } from "next/server";
import { loadCsv, fileExists } from "@/lib/csv";
import { getDataset } from "@/lib/datasets";
import { applyAugment } from "@/lib/server/augment";
import { toNum } from "@/lib/format";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const key = sp.get("key") ?? "";
  const spec = getDataset(key);
  if (!spec) {
    return NextResponse.json({ error: `Unknown dataset key: ${key}` }, { status: 404 });
  }
  if (!fileExists(spec.file)) {
    return NextResponse.json(
      { error: "data_not_available", file: spec.file, columns: [], rows: [], total: 0 },
      { status: 200 }
    );
  }

  let parsed;
  try {
    parsed = applyAugment(spec.augment, loadCsv(spec.file));
  } catch (e) {
    return NextResponse.json(
      { error: "parse_error", message: String(e), columns: [], rows: [], total: 0 },
      { status: 200 }
    );
  }

  const hide = new Set(spec.hide ?? []);
  const columns = parsed.columns.filter((c) => !hide.has(c));

  // ---- filter ----
  const q = (sp.get("q") ?? "").trim().toLowerCase();
  let rows = parsed.rows;
  if (q) {
    rows = rows.filter((r) =>
      columns.some((c) => String(r[c] ?? "").toLowerCase().includes(q))
    );
  }

  // ---- sort ----
  const sortCol = sp.get("sort") || spec.defaultSort?.col || columns[0];
  const dir = (sp.get("dir") || spec.defaultSort?.dir || "asc") === "desc" ? -1 : 1;
  if (sortCol && columns.includes(sortCol)) {
    rows = [...rows].sort((a, b) => {
      const av = a[sortCol];
      const bv = b[sortCol];
      const an = toNum(av);
      const bn = toNum(bv);
      const aNum = !Number.isNaN(an);
      const bNum = !Number.isNaN(bn);
      if (aNum && bNum) return (an - bn) * dir;
      if (aNum) return -1; // numbers before non-numbers
      if (bNum) return 1;
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    });
  }

  const total = rows.length;

  // ---- paginate ----
  const pageSize = Math.max(1, Math.min(200, Number(sp.get("pageSize")) || spec.pageSize || 15));
  const page = Math.max(1, Number(sp.get("page")) || 1);
  const start = (page - 1) * pageSize;
  const pageRows = rows.slice(start, start + pageSize).map((r) => {
    const o: Record<string, string> = {};
    for (const c of columns) o[c] = r[c] ?? "";
    return o;
  });

  return NextResponse.json({
    columns,
    rows: pageRows,
    total,
    page,
    pageSize,
    sort: sortCol,
    dir: dir === -1 ? "desc" : "asc",
    badgeCols: spec.badgeCols ?? [],
    labels: spec.labels ?? {},
  });
}
