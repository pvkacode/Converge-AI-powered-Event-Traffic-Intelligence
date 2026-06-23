export interface HeroStats {
  incidentsTotal: number;
  hotspotsSignificant: number;
  junctionsTotal: number;
  cvarReductionPct: number;
  criticalRetrainTriggers: number;
  healthCriticalChecks: number;
  healthTotalChecks: number;
  topSpilloverZone: string;
  spilloverPValue: number;
  spilloverPMantissa: number;
  spilloverPExponent: number;
  rsfCIndex: number;
  plannedEventMae: number;
  within20Pct: number;
  censoredRows: number;
  layerMetrics: Record<string, string>;
}
