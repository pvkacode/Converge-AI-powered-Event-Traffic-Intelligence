"use client";
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  Legend,
} from "recharts";
import { useTheme } from "./ThemeProvider";
import { fmtNum } from "@/lib/format";

// Read the live design-token colours so charts recolour on theme toggle.
export function useVizColors() {
  const { theme } = useTheme();
  const [c, setC] = useState({
    viz: ["#0F6E66", "#B5481F", "#8A6308", "#5B5440", "#6A4A7A"],
    ink: "#1A1813",
    ink3: "#8C8462",
    grid: "rgba(26,24,19,0.10)",
    accent: "#0F6E66",
    surface: "#FDF4CB",
  });
  useEffect(() => {
    const s = getComputedStyle(document.documentElement);
    const v = (n: string, f: string) => (s.getPropertyValue(n).trim() || f);
    setC({
      viz: [
        v("--viz-1", "#0F6E66"),
        v("--viz-2", "#B5481F"),
        v("--viz-3", "#8A6308"),
        v("--viz-4", "#5B5440"),
        v("--viz-5", "#6A4A7A"),
      ],
      ink: v("--ink", "#1A1813"),
      ink3: v("--ink-3", "#8C8462"),
      grid: v("--grid", "rgba(26,24,19,0.10)"),
      accent: v("--accent", "#0F6E66"),
      surface: v("--surface", "#FDF4CB"),
    });
  }, [theme]);
  return c;
}

function Tip({ active, payload, label, unit }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tip">
      {label != null && label !== "" && (
        <div className="tip-key" style={{ marginBottom: 4 }}>
          {label}
        </div>
      )}
      {payload.map((p: any, i: number) => (
        <div key={i} className="row gap-2" style={{ justifyContent: "space-between", gap: 16 }}>
          <span className="tip-key">{p.name}</span>
          <span className="tip-val" style={{ color: p.color || p.fill }}>
            {typeof p.value === "number" ? fmtNum(p.value) : p.value}
            {unit || ""}
          </span>
        </div>
      ))}
    </div>
  );
}

export function Legend2({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label} className="legend-item">
          <span className="legend-swatch" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

// Horizontal ranking bars. data: [{ name, value }]
export function HBar({
  data,
  height = 280,
  unit,
  colorIndex = 0,
  colorByValue,
}: {
  data: { name: string; value: number }[];
  height?: number;
  unit?: string;
  colorIndex?: number;
  colorByValue?: (v: number, i: number) => string;
}) {
  const c = useVizColors();
  return (
    <div className="chart-wrap" style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid horizontal={false} stroke={c.grid} />
          <XAxis type="number" stroke={c.ink3} tick={{ fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            stroke={c.ink3}
            tick={{ fontSize: 11, fill: c.ink }}
            interval={0}
          />
          <Tooltip content={<Tip unit={unit} />} cursor={{ fill: c.grid }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={22}>
            {data.map((d, i) => (
              <Cell key={i} fill={colorByValue ? colorByValue(d.value, i) : c.viz[colorIndex]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Vertical bars. data rows keyed by xKey + one numeric series.
export function VBar({
  data,
  xKey,
  yKey,
  height = 260,
  unit,
  colorIndex = 0,
}: {
  data: Record<string, any>[];
  xKey: string;
  yKey: string;
  height?: number;
  unit?: string;
  /** rows may carry a precomputed `__color` to colour individual bars */
  colorIndex?: number;
}) {
  const c = useVizColors();
  return (
    <div className="chart-wrap" style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke={c.grid} />
          <XAxis dataKey={xKey} stroke={c.ink3} tick={{ fontSize: 11 }} interval={0} angle={0} />
          <YAxis stroke={c.ink3} tick={{ fontSize: 11 }} />
          <Tooltip content={<Tip unit={unit} />} cursor={{ fill: c.grid }} />
          <Bar dataKey={yKey} radius={[4, 4, 0, 0]} maxBarSize={48}>
            {data.map((d, i) => (
              <Cell key={i} fill={(d.__color as string) || c.viz[colorIndex]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Multi-line chart. series: [{ key, label, colorIndex }]
export function LineSeries({
  data,
  xKey,
  series,
  height = 280,
  xLabel,
  yLabel,
  yDomain,
}: {
  data: Record<string, any>[];
  xKey: string;
  series: { key: string; label: string; colorIndex: number }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  yDomain?: [number, number];
}) {
  const c = useVizColors();
  return (
    <>
      <div className="chart-wrap" style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: xLabel ? 20 : 4, left: 0 }}>
            <CartesianGrid stroke={c.grid} />
            <XAxis
              dataKey={xKey}
              stroke={c.ink3}
              tick={{ fontSize: 11 }}
              label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: c.ink3, fontSize: 11 } : undefined}
            />
            <YAxis
              stroke={c.ink3}
              tick={{ fontSize: 11 }}
              domain={yDomain}
              label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: c.ink3, fontSize: 11 } : undefined}
            />
            <Tooltip content={<Tip />} />
            {series.map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                name={s.label}
                stroke={c.viz[s.colorIndex]}
                strokeWidth={2}
                dot={{ r: 2.5 }}
                activeDot={{ r: 4 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 8 }}>
        <Legend2 items={series.map((s) => ({ label: s.label, color: c.viz[s.colorIndex] }))} />
      </div>
    </>
  );
}

// Reliability / calibration diagram: predicted (x) vs observed (y) with the
// perfect-calibration diagonal as reference. series each a line of points.
export function ReliabilityChart({
  series,
  height = 300,
}: {
  series: { label: string; colorIndex: number; points: { x: number; y: number }[] }[];
  height?: number;
}) {
  const c = useVizColors();
  return (
    <>
      <div className="chart-wrap" style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 4 }}>
            <CartesianGrid stroke={c.grid} />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0, 1]}
              stroke={c.ink3}
              tick={{ fontSize: 11 }}
              label={{ value: "Predicted probability", position: "insideBottom", offset: -12, fill: c.ink3, fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              stroke={c.ink3}
              tick={{ fontSize: 11 }}
              label={{ value: "Observed frequency", angle: -90, position: "insideLeft", fill: c.ink3, fontSize: 11 }}
            />
            <ZAxis range={[40, 40]} />
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
              stroke={c.ink3}
              strokeDasharray="4 4"
              ifOverflow="extendDomain"
            />
            <Tooltip content={<Tip />} cursor={{ strokeDasharray: "3 3" }} />
            {series.map((s) => (
              <Scatter
                key={s.label}
                name={s.label}
                data={s.points}
                fill={c.viz[s.colorIndex]}
                line={{ stroke: c.viz[s.colorIndex], strokeWidth: 2 }}
                lineType="joint"
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 8 }}>
        <Legend2
          items={[
            ...series.map((s) => ({ label: s.label, color: c.viz[s.colorIndex] })),
            { label: "Perfect calibration", color: c.ink3 },
          ]}
        />
      </div>
    </>
  );
}

// Pareto / trade-off scatter. points [{ x, y }]
export function ParetoScatter({
  points,
  xLabel,
  yLabel,
  height = 300,
}: {
  points: { x: number; y: number; label?: string }[];
  xLabel: string;
  yLabel: string;
  height?: number;
}) {
  const c = useVizColors();
  return (
    <div className="chart-wrap" style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke={c.grid} />
          <XAxis
            type="number"
            dataKey="x"
            stroke={c.ink3}
            tick={{ fontSize: 11 }}
            label={{ value: xLabel, position: "insideBottom", offset: -12, fill: c.ink3, fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            stroke={c.ink3}
            tick={{ fontSize: 11 }}
            label={{ value: yLabel, angle: -90, position: "insideLeft", fill: c.ink3, fontSize: 11 }}
          />
          <ZAxis range={[60, 60]} />
          <Tooltip content={<Tip />} cursor={{ strokeDasharray: "3 3" }} />
          <Scatter
            data={points}
            fill={c.viz[0]}
            line={{ stroke: c.viz[0], strokeWidth: 1.5 }}
            lineType="joint"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// Grouped horizontal bars. data: [{ name, key1, key2, ... }]
export function GroupedHBar({
  data,
  keys,
  height = 320,
  xDomain = [0, 1] as [number, number],
  colors,
  rawByZone,
}: {
  data: Record<string, any>[];
  keys: { key: string; label: string; color: string }[];
  height?: number;
  xDomain?: [number, number];
  colors?: string[];
  rawByZone?: Record<string, Record<string, number>>;
}) {
  const c = useVizColors();
  const grid = "rgba(255,255,255,0.06)";

  const CustomTip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const raw = rawByZone?.[label as string];
    return (
      <div className="chart-tip">
        <div className="tip-key" style={{ marginBottom: 4 }}>{label}</div>
        {payload.map((p: any, i: number) => (
          <div key={i} className="row gap-2" style={{ justifyContent: "space-between", gap: 16 }}>
            <span className="tip-key">{p.name}</span>
            <span className="tip-val" style={{ color: p.color || p.fill }}>
              norm {typeof p.value === "number" ? fmtNum(p.value) : p.value}
              {raw && p.dataKey && raw[p.dataKey as string] != null
                ? ` · raw ${fmtNum(raw[p.dataKey as string])}`
                : ""}
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
      <div className="chart-wrap" style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid horizontal={false} stroke={grid} />
            <XAxis type="number" domain={xDomain} stroke={c.ink3} tick={{ fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={140}
              stroke={c.ink3}
              tick={{ fontSize: 11, fill: c.ink }}
              interval={0}
            />
            <Tooltip content={<CustomTip />} cursor={{ fill: c.grid }} />
            <Legend />
            {keys.map((k, i) => (
              <Bar
                key={k.key}
                dataKey={k.key}
                name={k.label}
                fill={colors?.[i] ?? k.color}
                radius={[0, 4, 4, 0]}
                maxBarSize={12}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 8 }}>
        <Legend2 items={keys.map((k, i) => ({ label: k.label, color: colors?.[i] ?? k.color }))} />
      </div>
    </>
  );
}

// Grouped before/after bars. data: [{ name, baseline, optimized }]
export function GroupedBar({
  data,
  keys,
  height = 280,
  unit,
}: {
  data: Record<string, any>[];
  keys: { key: string; label: string; colorIndex: number }[];
  height?: number;
  unit?: string;
}) {
  const c = useVizColors();
  return (
    <>
      <div className="chart-wrap" style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid vertical={false} stroke={c.grid} />
            <XAxis dataKey="name" stroke={c.ink3} tick={{ fontSize: 11 }} interval={0} />
            <YAxis stroke={c.ink3} tick={{ fontSize: 11 }} />
            <Tooltip content={<Tip unit={unit} />} cursor={{ fill: c.grid }} />
            {keys.map((k) => (
              <Bar key={k.key} dataKey={k.key} name={k.label} fill={c.viz[k.colorIndex]} radius={[4, 4, 0, 0]} maxBarSize={40} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: 8 }}>
        <Legend2 items={keys.map((k) => ({ label: k.label, color: c.viz[k.colorIndex] }))} />
      </div>
    </>
  );
}
