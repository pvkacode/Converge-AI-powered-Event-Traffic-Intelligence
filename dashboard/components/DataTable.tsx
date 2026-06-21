"use client";
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  MagnifyingGlass,
  CaretUp,
  CaretDown,
  CaretLeft,
  CaretRight,
  ArrowsDownUp,
  WarningCircle,
} from "@phosphor-icons/react";
import { fmtNum, humanize, isNumeric, titleCaseValue, toNum } from "@/lib/format";
import { ValueBadge, Note } from "./ui";

interface ApiResp {
  columns: string[];
  rows: Record<string, string>[];
  total: number;
  page: number;
  pageSize: number;
  sort: string;
  dir: "asc" | "desc";
  badgeCols: string[];
  labels: Record<string, string>;
  cellFlags?: { column: string; whenGt: number; badge: string; title: string }[];
  error?: string;
  message?: string;
  file?: string;
}

interface Props {
  dataset: string;
  title?: string;
  subtitle?: string;
  pageSize?: number;
  searchPlaceholder?: string;
  /** restrict to these columns (in order); omit to show all (minus hidden) */
  columns?: string[];
  /** pre-seed the free-text filter (used by the worked example) */
  initialQuery?: string;
  /** optional note rendered below the table title */
  headerNote?: React.ReactNode;
}

const PAGE_SIZES = [10, 15, 25, 50, 100];

export function DataTable({
  dataset,
  title,
  subtitle,
  pageSize: initialPageSize,
  searchPlaceholder = "Filter rows…",
  columns: only,
  initialQuery = "",
  headerNote,
}: Props) {
  const [q, setQ] = useState(initialQuery);
  const [debouncedQ, setDebouncedQ] = useState(initialQuery);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize ?? 15);
  const [sort, setSort] = useState<string | null>(null);
  const [dir, setDir] = useState<"asc" | "desc">("asc");
  const [data, setData] = useState<ApiResp | null>(null);
  const [loading, setLoading] = useState(true);
  const [errored, setErrored] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // debounce search input
  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedQ(q);
      setPage(1);
    }, 250);
    return () => clearTimeout(t);
  }, [q]);

  const fetchData = useCallback(async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setErrored(null);
    const params = new URLSearchParams({
      key: dataset,
      page: String(page),
      pageSize: String(pageSize),
    });
    if (debouncedQ) params.set("q", debouncedQ);
    if (sort) {
      params.set("sort", sort);
      params.set("dir", dir);
    }
    try {
      const res = await fetch(`/api/dataset?${params.toString()}`, { signal: ac.signal });
      if (!res.ok && res.status !== 200) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `Request failed (${res.status})`);
      }
      const json: ApiResp = await res.json();
      if (json.error === "data_not_available") {
        setData({ ...json, columns: [], rows: [], total: 0 } as ApiResp);
      } else if (json.error) {
        throw new Error(json.message || json.error);
      } else {
        setData(json);
        // adopt server defaults for sort on first load
        if (!sort && json.sort) {
          setSort(json.sort);
          setDir(json.dir);
        }
      }
    } catch (e: unknown) {
      if ((e as { name?: string })?.name === "AbortError") return;
      setErrored((e as Error).message || "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, [dataset, page, pageSize, debouncedQ, sort, dir]);

  useEffect(() => {
    fetchData();
    return () => abortRef.current?.abort();
  }, [fetchData]);

  const allColumns = data?.columns ?? [];
  const columns = useMemo(() => {
    if (only && only.length) return only.filter((c) => allColumns.includes(c));
    return allColumns;
  }, [only, allColumns]);

  const badgeSet = useMemo(() => new Set(data?.badgeCols ?? []), [data]);
  const labels = data?.labels ?? {};
  const cellFlagMap = useMemo(() => {
    const m = new Map<string, { whenGt: number; badge: string; title: string }>();
    for (const f of data?.cellFlags ?? []) m.set(f.column, f);
    return m;
  }, [data?.cellFlags]);

  // numeric column detection from the current page
  const numericCols = useMemo(() => {
    const set = new Set<string>();
    if (!data) return set;
    for (const c of columns) {
      if (badgeSet.has(c)) continue;
      const sample = data.rows.map((r) => r[c]).filter((v) => v !== "" && v != null);
      if (sample.length && sample.every((v) => isNumeric(v))) set.add(c);
    }
    return set;
  }, [data, columns, badgeSet]);

  const onSort = (col: string) => {
    if (sort === col) {
      setDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSort(col);
      // sensible default: numbers desc, text asc
      setDir(numericCols.has(col) ? "desc" : "asc");
    }
    setPage(1);
  };

  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);
  const colCount = Math.max(1, columns.length);

  return (
    <section className="panel">
      {(title || subtitle) && (
        <div className="panel-head">
          <div className="stack" style={{ gap: 2 }}>
            {title && <h2 className="section-title">{title}</h2>}
            {subtitle && <span className="section-meta">{subtitle}</span>}
            {headerNote ? <div style={{ marginTop: 10 }}>{headerNote}</div> : null}
          </div>
        </div>
      )}

      <div className="table-toolbar">
        <div className="field" style={{ minWidth: 240, flex: "0 1 320px" }}>
          <span className="field-icon">
            <MagnifyingGlass size={15} />
          </span>
          <input
            className="input with-icon"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={searchPlaceholder}
            aria-label="Filter rows"
          />
        </div>
        <div className="row gap-3">
          <span className="dim mono" style={{ fontSize: 12 }}>
            {loading ? "loading…" : `${total.toLocaleString()} rows`}
          </span>
          <select
            className="select"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            aria-label="Rows per page"
          >
            {PAGE_SIZES.map((n) => (
              <option key={n} value={n}>
                {n} / page
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="table-wrap">
        {errored ? (
          <div className="empty">
            <WarningCircle size={26} className="empty-icon" />
            <div className="empty-title">Could not load data</div>
            <div style={{ maxWidth: 420 }}>{errored}</div>
            <button className="btn btn-sm" onClick={fetchData} style={{ marginTop: 8 }}>
              Retry
            </button>
          </div>
        ) : !loading && total === 0 ? (
          <div className="empty">
            <div className="empty-title">
              {debouncedQ ? "No rows match your filter" : "Data not available"}
            </div>
            <div style={{ maxWidth: 420 }}>
              {debouncedQ
                ? `Nothing matched "${debouncedQ}". Clear the filter to see all rows.`
                : "This dataset is empty or the source file is missing from outputs/."}
            </div>
            {debouncedQ && (
              <button className="btn btn-sm" onClick={() => setQ("")} style={{ marginTop: 8 }}>
                Clear filter
              </button>
            )}
          </div>
        ) : (
          <table className="data">
            <thead>
              <tr>
                {columns.map((c) => {
                  const num = numericCols.has(c);
                  const sorted = sort === c;
                  return (
                    <th
                      key={c}
                      className={`${num ? "num" : ""} ${sorted ? "sorted" : ""}`}
                      onClick={() => onSort(c)}
                      title={`Sort by ${labels[c] || humanize(c)}`}
                      scope="col"
                    >
                      <span className="th-inner">
                        {labels[c] || humanize(c)}
                        {sorted ? (
                          dir === "asc" ? (
                            <CaretUp size={11} weight="bold" className="sort-caret" />
                          ) : (
                            <CaretDown size={11} weight="bold" className="sort-caret" />
                          )
                        ) : (
                          <ArrowsDownUp size={11} style={{ opacity: 0.32 }} />
                        )}
                      </span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: Math.min(pageSize, 10) }).map((_, i) => (
                    <tr key={i}>
                      <td colSpan={colCount} style={{ padding: 0 }}>
                        <div className="skel skel-row" />
                      </td>
                    </tr>
                  ))
                : data!.rows.map((r, i) => (
                    <tr key={i}>
                      {columns.map((c) => {
                        const raw = r[c] ?? "";
                        if (badgeSet.has(c)) {
                          return (
                            <td key={c}>
                              <ValueBadge value={raw} column={c} />
                            </td>
                          );
                        }
                        if (numericCols.has(c)) {
                          const flag = cellFlagMap.get(c);
                          const num = toNum(raw);
                          const showFlag = flag && !Number.isNaN(num) && num > flag.whenGt;
                          return (
                            <td
                              key={c}
                              className="num"
                              title={showFlag ? flag.title : undefined}
                            >
                              {raw === "" ? (
                                <span className="dim">-</span>
                              ) : (
                                <>
                                  {fmtNum(raw)}
                                  {showFlag ? (
                                    <span
                                      style={{
                                        fontSize: 10,
                                        background: "var(--warning-bg)",
                                        color: "var(--warning)",
                                        borderRadius: 4,
                                        padding: "1px 5px",
                                        marginLeft: 6,
                                        fontFamily: "var(--font-mono)",
                                        whiteSpace: "nowrap",
                                      }}
                                    >
                                      {flag.badge}
                                    </span>
                                  ) : null}
                                </>
                              )}
                            </td>
                          );
                        }
                        const long = raw.length > 46;
                        return (
                          <td key={c} className={long ? "td-truncate" : ""} title={long ? raw : undefined}>
                            {raw === "" ? <span className="dim">-</span> : raw}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="table-foot">
        <span>
          {total === 0 ? "0 results" : `${from.toLocaleString()}-${to.toLocaleString()} of ${total.toLocaleString()}`}
          {sort && !loading && (
            <span className="dim">
              {"  ·  sorted by "}
              <span className="mono">{labels[sort] || humanize(sort)}</span> {dir}
            </span>
          )}
        </span>
        <div className="pager">
          <button
            className="iconbtn"
            disabled={page <= 1 || loading}
            onClick={() => setPage(1)}
            aria-label="First page"
            title="First page"
          >
            «
          </button>
          <button
            className="iconbtn"
            disabled={page <= 1 || loading}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            aria-label="Previous page"
          >
            <CaretLeft size={15} />
          </button>
          <span className="mono" style={{ fontSize: 12, padding: "0 8px" }}>
            {page} / {totalPages}
          </span>
          <button
            className="iconbtn"
            disabled={page >= totalPages || loading}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            aria-label="Next page"
          >
            <CaretRight size={15} />
          </button>
          <button
            className="iconbtn"
            disabled={page >= totalPages || loading}
            onClick={() => setPage(totalPages)}
            aria-label="Last page"
            title="Last page"
          >
            »
          </button>
        </div>
      </div>
    </section>
  );
}
