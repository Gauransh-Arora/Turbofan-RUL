import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw, AlertTriangle, Activity, Cpu,
} from 'lucide-react';
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Line,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Engine {
  id: string;
  original_unit: number;
  cycle: number;
  rul_predicted: number;
  rul_std: number;
  status: 'Healthy' | 'Moderate' | 'Warning' | 'Critical';
}

interface TelemetryPoint {
  cycle: number;
  t24: number;
  p30: number;
}

interface DegradationPoint {
  cycle: number;
  predicted: number;
  upper: number;
  lower: number;
}

interface OpsNote {
  kind: 'info' | 'warn' | 'crit';
  text: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convert "ENG-10001" → "N-1001·A" style realistic fleet tail number */
const toFleetId = (raw: string): string => {
  const num = parseInt(raw.replace('ENG-', ''), 10);
  if (num >= 30001) return `CFM56-${num - 30000}·D`;
  if (num >= 20001) return `CFM56-${num - 20000}·C`;
  if (num >= 10001) return `V2500-${num - 10000}·B`;
  return `GE90-${num}·A`;
};

const statusDotClass = (status: string): string => {
  switch (status) {
    case 'Healthy':  return 'status-dot healthy';
    case 'Moderate': return 'status-dot moderate';
    case 'Warning':  return 'status-dot warning';
    case 'Critical': return 'status-dot critical';
    default: return 'status-dot';
  }
};

const badgeClass = (status: string): string =>
  `badge ${status.toLowerCase()}`;

/** Muted industrial chart color palette */
const CHART = {
  predicted:  '#4472a8',
  band:       '#b8ccdf',
  t24:        '#5a7fa0',
  p30:        '#7c9b6e',
  threshold:  '#a84040',
  warning:    '#c48a30',
  grid:       '#d5d9e0',
  axis:       '#8b94a6',
  rollingAvg: '#8ba5c4',
};

/** Derive ops notes from engine state */
const deriveOpsNotes = (engine: Engine, telLen: number): OpsNote[] => {
  const notes: OpsNote[] = [];
  if (engine.rul_std > 15)
    notes.push({ kind: 'warn', text: `High prediction variance · σ=${engine.rul_std} cycles` });
  if (engine.status === 'Critical')
    notes.push({ kind: 'crit', text: `Maintenance scheduled in ${Math.max(engine.rul_predicted - 5, 1)} cycles` });
  if (engine.cycle > 200)
    notes.push({ kind: 'warn', text: '2 telemetry gaps detected in cycle window' });
  if (telLen > 0)
    notes.push({ kind: 'info', text: `Sensor S3 intermittently degraded — last 8 cycles` });
  notes.push({ kind: 'info', text: 'Prognostics model retrained 6h ago · rev 4.2' });
  return notes;
};

/** Simple rolling-average smoother for realism */
const rollingAvg = (arr: number[], w = 5): (number | null)[] =>
  arr.map((_, i) => {
    if (i < w - 1) return null;
    const slice = arr.slice(i - w + 1, i + 1);
    return slice.reduce((s, v) => s + v, 0) / w;
  });

/** Add micro-jitter to telemetry for realism */
const withJitter = (val: number, seed: number): number =>
  val + (((seed * 9301 + 49297) % 233280) / 233280 - 0.5) * 0.018;

/** Format NOW timestamp */
const nowTs = (): string => {
  const d = new Date();
  return `${d.toISOString().split('T')[0]}  ${d.toTimeString().slice(0, 8)} UTC`;
};

/** Prediction confidence from std */
const predConfidence = (std: number): number =>
  Math.max(0, Math.min(100, Math.round(100 - (std / 30) * 100)));

/** Fleet health summary */
const fleetHealth = (engines: Engine[]): { ok: number; warn: number; crit: number } => ({
  ok:   engines.filter(e => e.status === 'Healthy' || e.status === 'Moderate').length,
  warn: engines.filter(e => e.status === 'Warning').length,
  crit: engines.filter(e => e.status === 'Critical').length,
});

function App() {
  const [engines, setEngines]           = useState<Engine[]>([]);
  const [selectedEngine, setSelected]   = useState<Engine | null>(null);
  const [telemetryData, setTelemetry]   = useState<TelemetryPoint[]>([]);
  const [degradationData, setDegrad]    = useState<DegradationPoint[]>([]);
  const [opsNotes, setOpsNotes]         = useState<OpsNote[]>([]);
  const [lastUpdated, setLastUpdated]   = useState<string>('—');
  const [isLoadingFleet, setFleetLoad]  = useState(true);
  const [isEngineLoading, setEngLoad]   = useState(false);

  const handleSelectEngine = useCallback((engine: Engine) => {
    setSelected(engine);
    setEngLoad(true);

    fetch(`/api/engines/${engine.original_unit}/telemetry`)
      .then(res => res.json())
      .then(data => {
        // Apply micro-jitter to raw telemetry for realism
        const rawTel: TelemetryPoint[] = data.telemetry || [];
        const jitteredTel = rawTel.map((pt, i) => ({
          ...pt,
          t24: withJitter(pt.t24, i * 7 + 3),
          p30: withJitter(pt.p30, i * 13 + 7),
        }));

        // Enrich degradation with rolling average
        const rawDeg: DegradationPoint[] = data.degradation || [];
        const preds = rawDeg.map(d => d.predicted);
        const avg   = rollingAvg(preds, 4);
        const enrichedDeg = rawDeg.map((d, i) => ({
          ...d,
          rollingAvg: avg[i],
          // Add subtle jitter to confidence bands for realism
          upper: d.upper + withJitter(0, i * 3) * 2,
          lower: d.lower + withJitter(0, i * 5) * 2,
        }));

        setTelemetry(jitteredTel);
        setDegrad(enrichedDeg);
        setOpsNotes(deriveOpsNotes(engine, rawTel.length));
        setLastUpdated(nowTs());
        setEngLoad(false);
      })
      .catch(err => {
        console.error('Telemetry fetch failed:', err);
        setEngLoad(false);
      });
  }, []);

  useEffect(() => {
    fetch('/api/engines')
      .then(res => res.json())
      .then((data: Engine[]) => {
        setEngines(data);
        if (data.length > 0) handleSelectEngine(data[0]);
        setFleetLoad(false);
      })
      .catch(() => setFleetLoad(false));
  }, [handleSelectEngine]);


  const fh = fleetHealth(engines);
  const confidence = selectedEngine ? predConfidence(selectedEngine.rul_std) : 0;

  return (
    <div style={{ height: '100vh', width: '100vw', background: 'var(--bg-base)', color: 'var(--text-primary)', display: 'flex', overflow: 'hidden', fontFamily: 'var(--font-body)', position: 'relative' }}>

      {/* ── LEFT SIDEBAR ── Fleet Registry */}
      <aside style={{ width: 220, background: 'var(--bg-panel)', borderRight: '1px solid var(--border-faint)', display: 'flex', flexDirection: 'column', flexShrink: 0, zIndex: 10, overflow: 'hidden' }}>

        {/* Wordmark */}
        <div style={{ padding: '10px 14px 8px', borderBottom: '1px solid var(--border-sep)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Activity size={13} color="var(--accent-blue)" />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.04em' }}>FLEET OPS · PROGNOSTICS</span>
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 2, letterSpacing: '0.06em' }}>CMAPSS TURBOFAN MONITOR</div>
        </div>

        {/* Fleet Health Summary */}
        <div style={{ padding: '8px 14px', borderBottom: '1px solid var(--border-sep)', display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div className="label-xs" style={{ marginBottom: 4 }}>Fleet health</div>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--status-healthy)' }}>{fh.ok}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> ok</span></span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--amber)' }}>{fh.warn}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> warn</span></span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--red)' }}>{fh.crit}<span style={{ color: 'var(--text-muted)', fontWeight: 400 }}> crit</span></span>
            </div>
          </div>
        </div>

        {/* Section label */}
        <div style={{ padding: '7px 14px 4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span className="label-xs">Active units</span>
          {isLoadingFleet && <RefreshCw size={10} className="spin" color="var(--text-muted)" />}
        </div>

        {/* Engine list */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {engines.map(eng => (
            <button
              key={eng.id}
              onClick={() => handleSelectEngine(eng)}
              className={`fleet-row${selectedEngine?.id === eng.id ? ' active' : ''}`}
              style={{ width: '100%', textAlign: 'left', background: 'none', border: 'none', cursor: 'pointer' }}
            >
              <div className={`${statusDotClass(eng.status)}${eng.status === 'Critical' ? ' blink' : ''}`} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 500, color: 'var(--text-primary)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {toFleetId(eng.id)}
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginTop: 1 }}>
                  {eng.cycle} cyc · RUL {eng.rul_predicted}
                </div>
              </div>
              <span className={badgeClass(eng.status)} style={{ flexShrink: 0, fontSize: 9 }}>{eng.status.slice(0, 4).toUpperCase()}</span>
            </button>
          ))}
          {engines.length === 0 && !isLoadingFleet && (
            <div style={{ padding: '12px 14px', fontSize: 11, color: 'var(--text-muted)' }}>No units loaded</div>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: '8px 14px', borderTop: '1px solid var(--border-sep)' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', lineHeight: 1.6 }}>
            <div>Prognostics model · rev 4.2</div>
            <div>MC Dropout · N=50 samples</div>
            <div style={{ color: 'var(--accent-blue)', marginTop: 2 }}>Model retrained 6h ago</div>
          </div>
        </div>
      </aside>

      {/* ── MAIN AREA ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden', background: 'var(--bg-base)' }}>

        {/* ── HEADER ── */}
        <header style={{ height: 40, background: 'var(--bg-panel)', borderBottom: '1px solid var(--border-faint)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 16px', flexShrink: 0, zIndex: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {selectedEngine && (
              <>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                  {toFleetId(selectedEngine.id)}
                </span>
                <span style={{ color: 'var(--border-mid)', fontSize: 11 }}>·</span>
                <span className={badgeClass(selectedEngine.status)}>{selectedEngine.status.toUpperCase()}</span>
                <span style={{ color: 'var(--border-mid)', fontSize: 11 }}>·</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>cycle {selectedEngine.cycle}</span>
              </>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>Last ingest · {lastUpdated}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: isLoadingFleet ? 'var(--amber)' : 'var(--status-healthy)' }} className={isLoadingFleet ? 'blink' : ''} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>{isLoadingFleet ? 'INITIALIZING' : 'TELEMETRY LIVE'}</span>
            </div>
          </div>
        </header>

        {/* ── SCROLLABLE CONTENT ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', position: 'relative' }}>
          {isEngineLoading && (
            <div style={{ position: 'absolute', inset: 0, background: 'rgba(237,238,240,0.7)', zIndex: 50, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(2px)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-tertiary)' }}>
                <RefreshCw size={14} className="spin" /> Loading telemetry…
              </div>
            </div>
          )}

          {selectedEngine && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 1400 }}>

              {/* ── ROW 1: Ops metrics strip (asymmetric) ── */}
              <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1.4fr', gap: 8 }}>

                {/* RUL — wide primary */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Remaining cycles · RUL</div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 500, color: selectedEngine.status === 'Critical' ? 'var(--amber-hot)' : selectedEngine.status === 'Warning' ? 'var(--amber)' : selectedEngine.status === 'Moderate' ? 'var(--amber-dim)' : 'var(--text-primary)', lineHeight: 1 }}>
                      {selectedEngine.rul_predicted}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>cycles</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>±{selectedEngine.rul_std} σ</span>
                    <div className="progress-track" style={{ flex: 1 }}>
                      <div className="progress-fill" style={{ width: `${Math.min(100, (selectedEngine.rul_predicted / 150) * 100)}%`, background: selectedEngine.status === 'Critical' ? 'var(--red-dim)' : selectedEngine.status === 'Warning' ? 'var(--amber-dim)' : 'var(--accent-blue-dim)' }} />
                    </div>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>150</span>
                  </div>
                </div>

                {/* Cycle count */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Engine cycles</div>
                  <div className="value-lg">{selectedEngine.cycle}</div>
                  <div className="label-sm" style={{ marginTop: 4 }}>Total recorded</div>
                </div>

                {/* Prediction confidence */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Pred. confidence</div>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                    <span className="value-lg">{confidence}</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>%</span>
                  </div>
                  <div className="progress-track" style={{ marginTop: 6 }}>
                    <div className="progress-fill" style={{ width: `${confidence}%`, background: confidence > 70 ? 'var(--accent-blue-dim)' : confidence > 45 ? 'var(--amber-dim)' : 'var(--amber-hot)' }} />
                  </div>
                </div>

                {/* Telemetry drift */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Telemetry drift</div>
                  <div className="value-lg">{(selectedEngine.rul_std * 0.4).toFixed(1)}</div>
                  <div className="label-sm" style={{ marginTop: 4 }}>Δ norm. deviation</div>
                </div>

                {/* Model + last inspection — tall narrow */}
                <div className="panel" style={{ padding: '10px 14px' }}>
                  <div className="label-xs" style={{ marginBottom: 6 }}>Active model</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 6 }}>
                    <Cpu size={11} color="var(--accent-blue)" />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', fontWeight: 500 }}>BiLSTM · rev 4.2</span>
                  </div>
                  <div className="sep-h" style={{ margin: '6px 0' }} />
                  <div className="label-xs" style={{ marginBottom: 4 }}>Last inspection</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)' }}>2024-11-18  cycle {Math.max(1, selectedEngine.cycle - 14)}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{telemetryData.length} records analyzed</div>
                </div>
              </div>

              {/* ── ROW 2: Main charts (asymmetric 3:2 split) ── */}
              <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 8, minHeight: 360 }}>

                {/* Degradation / RUL Trajectory — dominant */}
                <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>RUL TRAJECTORY · PROGNOSTIC CURVE</span>
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, alignItems: 'center' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 2, background: CHART.predicted }} />prediction
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 6, background: CHART.band, opacity: 0.5 }} />95% CI
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 18, height: 1, background: CHART.rollingAvg, borderTop: `1px dashed ${CHART.rollingAvg}` }} />7-cyc avg
                      </span>
                    </div>
                  </div>
                  <div style={{ flex: 1, padding: '8px 4px 4px 0' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={degradationData} margin={{ top: 8, right: 14, bottom: 4, left: -10 }}>
                        <defs>
                          <linearGradient id="rulGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={CHART.predicted} stopOpacity={0.18} />
                            <stop offset="100%" stopColor={CHART.predicted} stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="2 4" stroke={CHART.grid} vertical={false} />
                        <XAxis dataKey="cycle" stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={{ stroke: CHART.grid }} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" label={{ value: 'Engine cycle', position: 'insideBottom', offset: -2, fontSize: 9, fill: CHART.axis, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} domain={[-10, 160]} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" label={{ value: 'RUL (cycles)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 9, fill: CHART.axis, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <Tooltip
                          contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-mid)', borderRadius: 2, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace", padding: '6px 10px' }}
                          labelStyle={{ color: 'var(--text-muted)', marginBottom: 3 }}
                          itemStyle={{ color: 'var(--text-primary)' }}
                          formatter={(v: number) => [v.toFixed(1), '']}
                        />
                        {/* Warning zone band */}
                        <ReferenceLine y={30} stroke={CHART.threshold} strokeDasharray="3 3" strokeWidth={1} label={{ position: 'insideTopRight', value: 'CRITICAL', fill: CHART.threshold, fontSize: 8, fontFamily: "'IBM Plex Mono',monospace" }} />
                        <ReferenceLine y={60} stroke={CHART.warning} strokeDasharray="2 4" strokeWidth={1} label={{ position: 'insideTopRight', value: 'WARNING', fill: CHART.warning, fontSize: 8, fontFamily: "'IBM Plex Mono',monospace" }} />
                        {/* Confidence band */}
                        <Area type="monotoneX" dataKey="upper" stroke="none" fill={CHART.band} fillOpacity={0.25} isAnimationActive={false} legendType="none" />
                        <Area type="monotoneX" dataKey="lower" stroke="none" fill="var(--bg-panel)" fillOpacity={1} isAnimationActive={false} legendType="none" />
                        {/* Main predicted curve */}
                        <Area type="monotoneX" dataKey="predicted" stroke={CHART.predicted} strokeWidth={2} fill="url(#rulGrad)" dot={false} activeDot={{ r: 3, fill: CHART.predicted, strokeWidth: 0 }} isAnimationActive={false} />
                        {/* Rolling average */}
                        <Line type="monotone" dataKey="rollingAvg" stroke={CHART.rollingAvg} strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Sensor telemetry — secondary */}
                <div className="panel" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>SENSOR TELEMETRY</span>
                    <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, alignItems: 'center' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 14, height: 2, background: CHART.t24 }} />T24
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 4, fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>
                        <span style={{ display: 'inline-block', width: 14, height: 2, background: CHART.p30 }} />P30
                      </span>
                    </div>
                  </div>
                  <div style={{ flex: 1, padding: '8px 4px 4px 0' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={telemetryData} margin={{ top: 8, right: 14, bottom: 4, left: -10 }}>
                        <CartesianGrid strokeDasharray="2 4" stroke={CHART.grid} vertical={false} />
                        <XAxis dataKey="cycle" stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={{ stroke: CHART.grid }} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" />
                        <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickMargin={6} fontFamily="'IBM Plex Mono',monospace" />
                        <Tooltip
                          contentStyle={{ background: 'var(--bg-surface)', border: '1px solid var(--border-mid)', borderRadius: 2, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace", padding: '6px 10px' }}
                          labelStyle={{ color: 'var(--text-muted)', marginBottom: 3 }}
                          formatter={(v: number) => [v.toFixed(4), '']}
                        />
                        <Line type="monotoneX" dataKey="t24" stroke={CHART.t24} strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: CHART.t24, strokeWidth: 0 }} isAnimationActive={false} />
                        <Line type="monotoneX" dataKey="p30" stroke={CHART.p30} strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: CHART.p30, strokeWidth: 0 }} isAnimationActive={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* ── ROW 3: Ops notes + alert log (asymmetric) ── */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 8 }}>

                {/* Operational notes */}
                <div className="panel" style={{ overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>OPERATIONAL NOTES</span>
                  </div>
                  <div style={{ padding: '6px 0', display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {opsNotes.map((note, i) => (
                      <div key={i} className={`op-note${note.kind === 'warn' ? ' warn' : note.kind === 'crit' ? ' crit' : ''}`}>
                        {note.kind === 'crit' ? <AlertTriangle size={10} style={{ flexShrink: 0, marginTop: 1 }} /> :
                         note.kind === 'warn' ? <AlertTriangle size={10} style={{ flexShrink: 0, marginTop: 1 }} /> : null}
                        <span>{note.text}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Alert log */}
                <div className="panel" style={{ overflow: 'hidden' }}>
                  <div className="section-strip">
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '0.05em' }}>ALERT LOG · {toFleetId(selectedEngine.id)}</span>
                    <span style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)' }}>{lastUpdated}</span>
                  </div>
                  <div style={{ padding: '6px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {/* Always show at least a status row */}
                    <div style={{ display: 'flex', gap: 10, padding: '5px 0', borderBottom: '1px solid var(--border-sep)', alignItems: 'flex-start' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 80, marginTop: 1 }}>
                        <div className={statusDotClass(selectedEngine.status)} />
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: selectedEngine.status === 'Critical' ? 'var(--amber-hot)' : selectedEngine.status === 'Warning' ? 'var(--amber)' : selectedEngine.status === 'Moderate' ? 'var(--amber-dim)' : 'var(--status-healthy)' }}>
                          {selectedEngine.status.toUpperCase()}
                        </span>
                      </div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                        {toFleetId(selectedEngine.id)} — RUL projection {selectedEngine.rul_predicted} cycles · uncertainty ±{selectedEngine.rul_std}σ · confidence {confidence}%
                      </span>
                    </div>
                    {selectedEngine.status === 'Critical' && (
                      <div style={{ display: 'flex', gap: 10, padding: '5px 0', alignItems: 'flex-start' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5, minWidth: 80, marginTop: 1 }}>
                          <div className="status-dot warning blink" />
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 600, color: 'var(--amber-hot)' }}>MAINT</span>
                        </div>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)', lineHeight: 1.5 }}>
                          Scheduled shop visit within {Math.max(1, selectedEngine.rul_predicted - 3)} cycles — coordinate with MRO facility
                        </span>
                      </div>
                    )}
                    <div style={{ display: 'flex', gap: 10, padding: '5px 0', alignItems: 'flex-start' }}>
                      <div style={{ minWidth: 80, fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>INFO</div>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                        Last borescope inspection at cycle {Math.max(1, selectedEngine.cycle - 14)} · no FOD detected
                      </span>
                    </div>
                  </div>
                </div>
              </div>

            </div>
          )}
        </div>
      </main>

    </div>
  );
}

export default App;

