import { useState, useEffect } from 'react';
import { Activity, AlertTriangle, TrendingDown, RefreshCw, Cpu, Clock, CheckCircle2, ShieldAlert, BarChart3 } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

// --- Types ---
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

interface Alert {
  id: number;
  timestamp: string;
  level: string;
  message: string;
}

// --- Helpers ---
const getHealthColor = (status: string) => {
  switch (status) {
    case 'Healthy': return 'text-emerald-500';
    case 'Moderate': return 'text-amber-500';
    case 'Warning': return 'text-orange-500';
    case 'Critical': return 'text-red-500';
    default: return 'text-slate-500';
  }
};

const getHealthBg = (status: string) => {
  switch (status) {
    case 'Healthy': return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    case 'Moderate': return 'bg-amber-50 text-amber-700 border-amber-200';
    case 'Warning': return 'bg-orange-50 text-orange-700 border-orange-200';
    case 'Critical': return 'bg-red-50 text-red-700 border-red-200';
    default: return 'bg-slate-50 text-slate-700 border-slate-200';
  }
};

const StatusIcon = ({ status, className }: { status: string, className?: string }) => {
  switch (status) {
    case 'Healthy': return <CheckCircle2 className={className} />;
    case 'Moderate': return <AlertTriangle className={className} />;
    case 'Warning': return <ShieldAlert className={className} />;
    case 'Critical': return <AlertTriangle className={className} />;
    default: return <Activity className={className} />;
  }
};

function App() {
  const [engines, setEngines] = useState<Engine[]>([]);
  const [selectedEngine, setSelectedEngine] = useState<Engine | null>(null);
  const [telemetryData, setTelemetryData] = useState<TelemetryPoint[]>([]);
  const [degradationData, setDegradationData] = useState<DegradationPoint[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  
  const [isLoadingFleet, setIsLoadingFleet] = useState(true);
  const [isEngineLoading, setIsEngineLoading] = useState(false);

  useEffect(() => {
    fetch('/api/engines')
      .then(res => res.json())
      .then((data: Engine[]) => {
        setEngines(data);
        if (data.length > 0) {
          handleSelectEngine(data[0]);
        }
        setIsLoadingFleet(false);
      })
      .catch(err => {
        console.error("Failed to fetch engines:", err);
        setIsLoadingFleet(false);
      });
  }, []);

  const handleSelectEngine = (engine: Engine) => {
    setSelectedEngine(engine);
    setIsEngineLoading(true);
    
    fetch(`/api/engines/${engine.original_unit}/telemetry`)
      .then(res => res.json())
      .then(data => {
        setTelemetryData(data.telemetry || []);
        setDegradationData(data.degradation || []);
        
        // Generate contextual alerts based on status
        if (engine.status !== 'Healthy') {
          setAlerts([{
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            level: engine.status,
            message: `Engine ${engine.id} flagged. RUL projected at ${engine.rul_predicted} cycles (±${engine.rul_std}).`
          }]);
        } else {
          setAlerts([{
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            level: 'Healthy',
            message: `Engine ${engine.id} is operating nominally within expected parameters.`
          }]);
        }
        
        setIsEngineLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch telemetry:", err);
        setIsEngineLoading(false);
      });
  };

  return (
    <div className="h-screen w-screen bg-[#f8f9fa] text-slate-800 flex overflow-hidden font-sans selection:bg-blue-100">
      
      {/* Left Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 z-10">
        <div className="h-16 flex items-center px-6 border-b border-slate-100">
          <div className="flex items-center gap-2 text-blue-600">
            <Activity className="w-5 h-5" />
            <span className="font-bold text-slate-900 text-lg tracking-tight">Aerospace RUL</span>
          </div>
        </div>
        
        <div className="p-4">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-2 flex justify-between items-center">
            <span>Active Fleet</span>
            {isLoadingFleet && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
          </div>
          
          <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-8rem)]">
            {engines.map(engine => (
              <button 
                key={engine.id} 
                onClick={() => handleSelectEngine(engine)}
                className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors flex items-center justify-between group
                  ${selectedEngine?.id === engine.id 
                    ? 'bg-blue-50 text-blue-700' 
                    : 'hover:bg-slate-50 text-slate-600'}`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full ${engine.status === 'Healthy' ? 'bg-emerald-500' : engine.status === 'Critical' ? 'bg-red-500' : 'bg-amber-500'}`}></div>
                  <span className={`font-medium text-sm ${selectedEngine?.id === engine.id ? 'text-blue-700' : 'text-slate-700'}`}>
                    {engine.id}
                  </span>
                </div>
                <span className="text-xs font-medium text-slate-400">
                  {engine.cycle}c
                </span>
              </button>
            ))}
            {engines.length === 0 && !isLoadingFleet && (
              <div className="px-3 py-2 text-sm text-slate-400 text-center">No engines</div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative bg-[#f8f9fa]">
        {/* Dot Background */}
        <div className="absolute inset-0 z-0 [background-size:20px_20px] [background-image:radial-gradient(#000000_1px,transparent_1px)]" />
        <div className="pointer-events-none absolute inset-0 z-0 flex items-center justify-center bg-[#f8f9fa] [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)]"></div>

        {/* Header */}
        <header className="h-16 bg-white/50 backdrop-blur-sm flex items-center justify-between px-8 shrink-0 z-10 relative">
          <h1 className="text-xl font-bold text-slate-800">Overview</h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-white border border-slate-200 px-3 py-1.5 rounded-md shadow-sm">
              <div className={`w-2 h-2 rounded-full ${isLoadingFleet ? 'bg-amber-400 animate-pulse' : 'bg-emerald-500'}`}></div>
              <span className="text-xs font-medium text-slate-600">{isLoadingFleet ? 'Initializing...' : 'System Nominal'}</span>
            </div>
            <div className="bg-white border border-slate-200 px-3 py-1.5 rounded-md shadow-sm text-xs font-medium text-slate-600 flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              {new Date().toISOString().split('T')[0]}
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 relative z-10">
          {isEngineLoading && (
            <div className="absolute inset-0 bg-white/40 z-50 flex items-center justify-center backdrop-blur-[2px]">
              <RefreshCw className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
          )}
          
          {selectedEngine && (
            <div className="flex flex-col gap-6 max-w-7xl mx-auto">
              
              {/* Top Metrics Row - 4 Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                
                {/* Metric 1: RUL */}
                <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-slate-500">Predicted RUL</span>
                    <TrendingDown className="w-4 h-4 text-slate-400" />
                  </div>
                  <div className="text-3xl font-bold text-slate-800 mb-1">{selectedEngine.rul_predicted}</div>
                  <div className="text-xs font-medium text-blue-600 mt-auto flex items-center gap-1">
                    <span className="bg-blue-50 px-1.5 py-0.5 rounded">± {selectedEngine.rul_std} cycles</span>
                  </div>
                </div>

                {/* Metric 2: Status */}
                <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-slate-500">Current State</span>
                    <StatusIcon status={selectedEngine.status} className={`w-4 h-4 ${getHealthColor(selectedEngine.status)}`} />
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1 capitalize">{selectedEngine.status}</div>
                  <div className={`text-xs font-medium mt-auto w-fit px-2 py-0.5 rounded-full border ${getHealthBg(selectedEngine.status)}`}>
                    Status Indicator
                  </div>
                </div>

                {/* Metric 3: Current Cycle */}
                <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-slate-500">Current Cycle</span>
                    <BarChart3 className="w-4 h-4 text-slate-400" />
                  </div>
                  <div className="text-3xl font-bold text-slate-800 mb-1">{selectedEngine.cycle}</div>
                  <div className="text-xs font-medium text-slate-500 mt-auto">
                    Total recorded cycles
                  </div>
                </div>

                {/* Metric 4: Model Info */}
                <div className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-slate-500">Active Model</span>
                    <Cpu className="w-4 h-4 text-slate-400" />
                  </div>
                  <div className="text-2xl font-bold text-slate-800 mb-1">BiLSTM V4</div>
                  <div className="text-xs font-medium text-slate-500 mt-auto">
                    {telemetryData.length} records analyzed
                  </div>
                </div>

              </div>

              {/* Visualizations Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-[380px]">
                
                {/* Chart 1: Telemetry (Blue & Green lines) */}
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-base font-semibold text-slate-800">Sensor Telemetry Trend</h3>
                    <div className="flex items-center gap-4 text-xs font-medium">
                      <div className="flex items-center gap-1.5 text-slate-600">
                        <div className="w-2 h-2 rounded-full bg-blue-500"></div> T24
                      </div>
                      <div className="flex items-center gap-1.5 text-slate-600">
                        <div className="w-2 h-2 rounded-full bg-emerald-400"></div> P30
                      </div>
                    </div>
                  </div>
                  <div className="flex-1 w-full min-h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={telemetryData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                        <XAxis dataKey="cycle" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickMargin={12} />
                        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickMargin={12} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '13px', fontWeight: 500 }}
                          itemStyle={{ color: '#0f172a' }}
                          labelStyle={{ color: '#64748b', marginBottom: '4px' }}
                        />
                        <Line type="monotone" dataKey="t24" stroke="#3b82f6" strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 0, fill: '#3b82f6' }} isAnimationActive={false} />
                        <Line type="monotone" dataKey="p30" stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 0, fill: '#10b981' }} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 2: Degradation (Area chart) */}
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-base font-semibold text-slate-800">Degradation Curve</h3>
                    <div className="text-xs font-medium text-slate-500 bg-slate-50 px-2 py-1 rounded">RUL Trajectory</div>
                  </div>
                  <div className="flex-1 w-full min-h-[250px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={degradationData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                        <defs>
                          <linearGradient id="colorDegradation" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                        <XAxis dataKey="cycle" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} tickMargin={12} />
                        <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} domain={[-20, 160]} tickMargin={12} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', fontSize: '13px', fontWeight: 500 }}
                          itemStyle={{ color: '#0f172a' }}
                          labelStyle={{ color: '#64748b', marginBottom: '4px' }}
                        />
                        <Area type="monotone" dataKey="upper" stroke="none" fill="#f8fafc" fillOpacity={0.5} isAnimationActive={false} />
                        <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={1} isAnimationActive={false} />
                        <Area type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2.5} fill="url(#colorDegradation)" dot={false} activeDot={{ r: 4, fill: '#3b82f6', strokeWidth: 0 }} isAnimationActive={false} />
                        <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Failure Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 500 }} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>
              
              {/* Alerts Row at the bottom to better fit the layout */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col">
                 <div className="flex items-center px-6 py-4 border-b border-slate-100">
                    <h3 className="text-base font-semibold text-slate-800">System Alerts</h3>
                 </div>
                 <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {alerts.map(alert => (
                    <div key={alert.id} className="bg-slate-50 rounded-lg p-4 flex gap-3 border border-slate-100">
                      <div className="mt-0.5">
                        <StatusIcon status={alert.level} className={`w-5 h-5 ${getHealthColor(alert.level)}`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <span className={`text-xs font-bold uppercase tracking-wider ${getHealthColor(alert.level)}`}>{alert.level}</span>
                          <span className="text-xs font-medium text-slate-400">{alert.timestamp}</span>
                        </div>
                        <p className="text-sm text-slate-700 leading-snug">{alert.message}</p>
                      </div>
                    </div>
                  ))}
                  {alerts.length === 0 && (
                    <div className="col-span-full py-6 text-center text-sm text-slate-500">
                      No active system alerts.
                    </div>
                  )}
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

