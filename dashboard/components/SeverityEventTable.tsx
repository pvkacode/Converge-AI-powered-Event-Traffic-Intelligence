"use client";
// Small, self-contained table for Layer 6 monitoring logs (retrain triggers,
// active alerts): severity filter chips, free-text search, click-to-sort
// columns, and an expandable detail row. Built directly on the same CSS
// classes/components the generic DataTable uses (table.data, .table-toolbar,
// EmptyState, Badge) rather than the paginated /api/dataset path, since these
// logs are small enough to filter/sort entirely client-side from props.
import { Fragment, useMemo, useState } from "react";
import { CaretDown, CaretRight, MagnifyingGlass } from "@phosphor-icons/react";
import { Badge, EmptyState } from "./ui";
import { fmtNum, humanize, isNumeric, toNum } from "@/lib/format";
import { severityRank, severityVariant } from "@/lib/severity";

export type Row = Record<string, string>;

export interface ColumnSpec {
  key: string;
  label?: string;
}

interface Props {
  rows: Row[];
  idField: string;
  severityField?: string;
  /** optional secondary sort key (numeric, descending) used as a tiebreak within a severity */
  scoreField?: string;
  columns: ColumnSpec[];
  detailFields: ColumnSpec[];
  searchFields: string[];
  searchPlaceholder?: string;
  emptyMessage?: string;
}

const SEVERITY_FILTERS = [
  { key: "all", label: "All" },
  { key: "critical", label: "Critical" },
  { key: "moderate", label: "Moderate" },
  { key: "info", label: "Info" },
] as const;

export function SeverityEventTable({
  rows,
  idField,
  severityField = "severity",
  scoreField,
  columns,
  detailFields,
  searchFields,
  searchPlaceholder = "Filter rows…",
  emptyMessage = "This dataset is empty or the source file is missing from outputs/.",
}: Props) {
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [q, setQ] = useState("");
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const counts = useMemo(() => {
    const c: Record<string, number> = { critical: 0, moderate: 0, info: 0 };
    for (const r of rows) {
      const s = (r[severityField] ?? "").trim().toLowerCase();
      if (s in c) c[s] += 1;
    }
    return c;
  }, [rows, severityField]);

  const filtered = useMemo(() => {
    let out = rows;
    if (severityFilter !== "all") {
      out = out.filter((r) => (r[severityField] ?? "").trim().toLowerCase() === severityFilter);
    }
    const query = q.trim().toLowerCase();
    if (query) {
      out = out.filter((r) => searchFields.some((f) => (r[f] ?? "").toLowerCase().includes(query)));
    }
    return out;
  }, [rows, severityFilter, q, searchFields, severityField]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    copy.sort((a, b) => {
      if (sortKey) {
        const av = a[sortKey] ?? "";
        const bv = b[sortKey] ?? "";
        const diff =
          isNumeric(av) && isNumeric(bv) ? toNum(av) - toNum(bv) : av.localeCompare(bv);
        return sortDir === "asc" ? diff : -diff;
      }
      // Default: severity priority (critical first), then score descending within a tier.
      const rankDiff = severityRank(a[severityField] ?? "") - severityRank(b[severityField] ?? "");
      if (rankDiff !== 0) return rankDiff;
      return scoreField ? toNum(b[scoreField]) - toNum(a[scoreField]) : 0;
    });
    return copy;
  }, [filtered, sortKey, sortDir, severityField, scoreField]);

  const onSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDir(key === severityField ? "asc" : isNumeric(rows[0]?.[key] ?? "") ? "desc" : "asc");
  };

  const toggleExpand = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (rows.length === 0) {
    return <EmptyState message={emptyMessage} />;
  }

  return (
    <div>
      <div className="table-toolbar">
        <div className="row gap-3" style={{ flexWrap: "wrap" }}>
          <div className="field" style={{ minWidth: 220 }}>
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
          <div className="row gap-2" style={{ flexWrap: "wrap" }}>
            {SEVERITY_FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                className={`btn btn-sm${severityFilter === f.key ? " btn-accent" : ""}`}
                onClick={() => setSeverityFilter(f.key)}
              >
                {f.label}
                {f.key !== "all" && (
                  <span className="dim mono" style={{ marginLeft: 4 }}>
                    {counts[f.key] ?? 0}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
        <span className="dim mono" style={{ fontSize: 12 }}>
          {sorted.length.toLocaleString()} rows
        </span>
      </div>

      <div className="table-wrap">
        {sorted.length === 0 ? (
          <EmptyState
            title="No rows match your filter"
            message={`Nothing matched the current filters. Clear them to see all ${rows.length} rows.`}
          />
        ) : (
          <table className="data">
            <thead>
              <tr>
                <th style={{ width: 28 }} />
                {columns.map((c) => {
                  const isSorted = sortKey ? sortKey === c.key : c.key === severityField;
                  return (
                    <th
                      key={c.key}
                      className={isSorted ? "sorted" : ""}
                      onClick={() => onSort(c.key)}
                      title={`Sort by ${c.label ?? humanize(c.key)}`}
                      scope="col"
                    >
                      <span className="th-inner">{c.label ?? humanize(c.key)}</span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const id = r[idField] || `${columns.map((c) => r[c.key]).join("|")}`;
                const isOpen = expanded.has(id);
                return (
                  <Fragment key={id}>
                    <tr onClick={() => toggleExpand(id)} style={{ cursor: "pointer" }}>
                      <td>{isOpen ? <CaretDown size={13} /> : <CaretRight size={13} />}</td>
                      {columns.map((c) => {
                        const raw = r[c.key] ?? "";
                        if (c.key === severityField) {
                          return (
                            <td key={c.key}>
                              <Badge variant={severityVariant(raw)}>{raw || "-"}</Badge>
                            </td>
                          );
                        }
                        if (raw !== "" && isNumeric(raw)) {
                          return (
                            <td key={c.key} className="num">
                              {fmtNum(raw)}
                            </td>
                          );
                        }
                        const long = raw.length > 46;
                        return (
                          <td key={c.key} className={long ? "td-truncate" : ""} title={long ? raw : undefined}>
                            {raw === "" ? <span className="dim">-</span> : raw}
                          </td>
                        );
                      })}
                    </tr>
                    {isOpen && (
                      <tr>
                        <td />
                        <td colSpan={columns.length} style={{ background: "var(--surface-inset)" }}>
                          <div className="stack gap-2" style={{ padding: "6px 0" }}>
                            {detailFields.map((f) => (
                              <div className="metric-line" key={f.key}>
                                <span className="ml-k">{f.label ?? humanize(f.key)}</span>
                                <span className="ml-v">{r[f.key] || "-"}</span>
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
