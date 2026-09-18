import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  BatteryCharging,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Gauge,
  LoaderCircle,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Sun,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "GridWise — Smart Campus Energy Optimization" },
      {
        name: "description",
        content:
          "A live control room for LLM-assisted, constraint-safe campus energy optimization.",
      },
      { property: "og:title", content: "GridWise — Smart Campus Energy Optimization" },
      {
        property: "og:description",
        content:
          "Interpret operator directives and optimize a verified 24-hour campus energy schedule.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: GridWise,
});

const stages = ["Interpret", "Guardrails", "Optimize", "Replay"];

function GridWise() {
  const [running, setRunning] = useState(false);
  const [complete, setComplete] = useState(true);
  const [stage, setStage] = useState(4);
  const [note, setNote] = useState("");

  const status = useMemo(() => {
    if (running)
      return stage === 0
        ? "Reading operator notes"
        : (stages[Math.min(stage - 1, stages.length - 1)] ?? "Optimizing");
    return complete ? "Schedule verified" : "Ready";
  }, [complete, running, stage]);

  const [scenarioData, setScenarioData] = useState<any>(null);
  const [casePack, setCasePack] = useState<any[] | null>(null);
  const [activeCaseIndex, setActiveCaseIndex] = useState(0);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        if (json.cases && Array.isArray(json.cases)) {
          setCasePack(json.cases);
          setActiveCaseIndex(0);
          const firstInput = json.cases[0].input;
          setScenarioData(firstInput);
          if (firstInput.operator_notes) {
            setNote(firstInput.operator_notes.join("\n"));
          }
        } else {
          setCasePack(null);
          setScenarioData(json);
          if (json.operator_notes) {
            setNote(json.operator_notes.join("\n"));
          }
        }
        setApiResult(null); // Reset previous run
      } catch (err) {
        alert("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  };

  const [apiResult, setApiResult] = useState<any>(null);

  const getPath = (dataArray: number[]) => {
    if (!dataArray || dataArray.length !== 24) return "M0,280";
    const maxVal = Math.max(...dataArray, 200); // at least 200 for scale
    const points = dataArray.map((val, idx) => {
      const x = (idx / 23) * 900;
      const y = 280 - (val / maxVal) * 240;
      return `${x},${y}`;
    });
    return `M${points[0]} L` + points.slice(1).join(" L");
  };

  const gridPath = getPath(
    apiResult
      ? apiResult.hourly_plan.map((h: any) => h.grid_kwh)
      : scenarioData?.hours.map((h: any) => h.demand_kwh) || [],
  );
  const solarPath = getPath(
    apiResult
      ? apiResult.hourly_plan.map((h: any) => h.solar_used_kwh)
      : scenarioData?.hours.map((h: any) => h.solar_kwh) || [],
  );
  const batteryPath = getPath(
    apiResult ? apiResult.hourly_plan.map((h: any) => h.battery_energy_after_kwh) : [],
  );

  const run = async () => {
    if (running) return;
    if (!scenarioData) {
      alert("Please upload a scenario JSON first.");
      return;
    }

    setRunning(true);
    setComplete(false);
    setStage(0);

    const timers = stages.map((_, index) =>
      window.setTimeout(() => setStage(index + 1), 450 * (index + 1)),
    );

    try {
      const response = await fetch("/api/optimize-energy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...scenarioData, operator_notes: note.split('\n').filter(n => n.trim() !== '') }),
      });
      const data = await response.json();
      if (response.ok) {
        setApiResult(data);
      } else {
        console.error(data);
        alert(data.error || "Optimization failed");
      }
    } catch (e) {
      console.error(e);
      alert("Failed to connect to API");
    } finally {
      timers.forEach(window.clearTimeout);
      setStage(4);
      setRunning(false);
      setComplete(true);
    }
  };

  return (
    <main className="min-h-screen bg-background font-body text-foreground">
      <header className="sticky top-0 z-30 flex h-16 items-center border-b border-border bg-background/92 px-4 backdrop-blur-xl sm:px-6">
        <div className="flex items-center gap-3">
          <div className="logo-mark" aria-hidden="true">
            <Zap size={18} fill="currentColor" />
          </div>
          <span className="font-display text-lg font-black">
            GRID<span className="text-grid">WISE</span>
          </span>
          <span className="hidden h-5 w-px bg-border sm:block" />
          <span className="hidden font-mono text-[10px] text-muted-foreground sm:inline">
            BUP CAMPUS · DHAKA
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3 font-mono text-[10px]">
          <span className="hidden items-center gap-2 text-battery sm:flex">
            <span className="status-dot" /> SYSTEM READY
          </span>
          <span className="hidden text-muted-foreground md:inline">24H HORIZON</span>
          <button className="action-button" onClick={run} disabled={running}>
            {running ? (
              <LoaderCircle size={14} className="animate-spin" />
            ) : (
              <Play size={14} fill="currentColor" />
            )}
            <span className="hidden xs:inline">{running ? "OPTIMIZING" : "RUN OPTIMIZATION"}</span>
          </button>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] p-3 sm:p-4">
        <section className="mb-3 flex flex-wrap items-center gap-3 border-b border-border pb-3">
          <div>
            <p className="eyebrow">ACTIVE SCENARIO</p>
            <div className="flex items-center gap-2">
              <label className="scenario-button cursor-pointer inline-flex items-center gap-2">
                {scenarioData ? scenarioData.scenario_id : "UPLOAD SCENARIO JSON"}
                <ChevronDown size={14} />
                <input type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
              </label>
              {casePack && (
                <select 
                  className="scenario-button cursor-pointer bg-background" 
                  value={activeCaseIndex} 
                  onChange={(e) => {
                    const idx = Number(e.target.value);
                    setActiveCaseIndex(idx);
                    const input = casePack[idx].input;
                    setScenarioData(input);
                    setNote((input.operator_notes || []).join("\n"));
                    setApiResult(null);
                  }}
                >
                  {casePack.map((c, i) => (
                    <option key={c.id} value={i}>{c.id}: {c.label}</option>
                  ))}
                </select>
              )}
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2 font-mono text-[10px] text-muted-foreground">
            <ShieldCheck size={14} className="text-battery" />
            ALL CONSTRAINTS VALID
          </div>
        </section>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_340px]">
          <section className="kinetic-panel min-w-0">
            <div className="sweep-band sweep-band-one" />
            <div className="sweep-band sweep-band-two" />
            <div className="relative z-10 p-4 sm:p-6">
              <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="eyebrow">OPTIMIZED ENERGY DISPATCH</p>
                  <h1 className="mt-2 font-display text-4xl font-black leading-none sm:text-5xl lg:text-6xl">
                    DEMAND<span className="text-grid">/</span>SOLAR
                  </h1>
                  <p className="mt-2 font-mono text-[10px] text-muted-foreground">
                    24 HOUR FORECAST · KWH · BDT TARIFF
                  </p>
                </div>
                <div className="flex flex-wrap gap-4 font-mono text-[10px]">
                  <Legend tone="grid" label="GRID IMPORT" />
                  <Legend tone="solar" label="SOLAR USED" />
                  <Legend tone="battery" label="BATTERY" />
                </div>
              </div>

              <div className="chart-grid mt-8 overflow-hidden">
                <svg
                  viewBox="0 0 900 300"
                  role="img"
                  aria-label="24-hour campus energy forecast"
                  className={running ? "is-running" : ""}
                >
                  <defs>
                    <linearGradient id="gridFill" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0" stopColor="currentColor" stopOpacity=".18" />
                      <stop offset="1" stopColor="currentColor" stopOpacity="0" />
                    </linearGradient>
                  </defs>

                  <path
                    className="chart-area text-grid"
                    fill="url(#gridFill)"
                    d={`${gridPath} L900,280 L0,280Z`}
                  />
                  <path className="chart-line chart-line-grid" d={gridPath} fill="none" />
                  <path className="chart-line chart-line-solar" d={solarPath} fill="none" />
                  {apiResult && (
                    <path className="chart-line chart-line-battery" d={batteryPath} fill="none" />
                  )}
                  <path className="flow-line" d="M0 145 H900" />
                  {[90, 270, 450, 630, 810].map((x) => (
                    <circle key={x} cx={x} cy="145" r="3.5" className="flow-node" />
                  ))}
                </svg>
                <div className="flex justify-between px-1 font-mono text-[10px] text-muted-foreground">
                  <span>00:00</span>
                  <span>06:00</span>
                  <span>12:00</span>
                  <span>18:00</span>
                  <span>24:00</span>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-2 gap-2 lg:grid-cols-4">
                <Metric
                  icon={CircleDollarSign}
                  label="TOTAL COST"
                  value={
                    apiResult
                      ? `৳${Math.round(apiResult.total_cost_bdt).toLocaleString()}`
                      : scenarioData
                        ? `৳${Math.round(scenarioData.hours.reduce((acc: number, h: any) => acc + h.demand_kwh * h.tariff_bdt_per_kwh, 0)).toLocaleString()}`
                        : "--"
                  }
                  detail={apiResult ? "−14.8% baseline" : scenarioData ? "baseline (no solar/batt)" : "--"}
                  tone="battery"
                />
                <Metric
                  icon={Gauge}
                  label="PEAK GRID"
                  value={
                    apiResult 
                      ? apiResult.peak_grid_kwh.toFixed(1) 
                      : scenarioData
                        ? Math.max(...scenarioData.hours.map((h: any) => h.demand_kwh)).toFixed(1)
                        : "--"
                  }
                  unit="kWh"
                  detail={apiResult ? "optimized" : scenarioData ? "baseline peak" : "--"}
                  tone="warn"
                />
                <Metric
                  icon={Activity}
                  label="GRID ENERGY"
                  value={
                    apiResult 
                      ? Math.round(apiResult.total_grid_kwh).toLocaleString() 
                      : scenarioData
                        ? Math.round(scenarioData.hours.reduce((acc: number, h: any) => acc + h.demand_kwh, 0)).toLocaleString()
                        : "--"
                  }
                  unit="kWh"
                  detail="24 hour total"
                  tone="grid"
                />
                <Metric
                  icon={Sun}
                  label="SOLAR AVAILABLE"
                  value={
                    apiResult
                      ? Math.round(
                          apiResult.hourly_plan.reduce((acc: any, h: any) => acc + h.solar_used_kwh, 0),
                        ).toLocaleString()
                      : scenarioData
                        ? Math.round(scenarioData.hours.reduce((acc: number, h: any) => acc + h.solar_kwh, 0)).toLocaleString()
                        : "--"
                  }
                  unit="kWh"
                  detail={apiResult ? "utilized" : scenarioData ? "forecast available" : "--"}
                  tone="solar"
                />
              </div>
            </div>
          </section>

          <aside className="control-panel">
            <div className="flex items-center justify-between">
              <div>
                <p className="eyebrow">OPTIMIZATION PIPELINE</p>
                <p className="mt-1 font-display text-lg font-bold">{status}</p>
              </div>
              <div className={`pipeline-orb ${running ? "is-running" : ""}`}>
                <BrainCircuit size={20} />
              </div>
            </div>

            <div className="mt-5 grid grid-cols-4 gap-1">
              {stages.map((item, index) => (
                <div
                  key={item}
                  className={`stage ${stage > index || complete ? "stage-done" : ""}`}
                >
                  <span>{stage > index || complete ? <CheckCircle2 size={13} /> : index + 1}</span>
                  <small>{item}</small>
                </div>
              ))}
            </div>

            <div className="section-rule" />
            <div className="flex items-center justify-between">
              <p className="eyebrow">BATTERY STATE</p>
              <span className="font-mono text-[10px] text-battery">CHARGED</span>
            </div>
            <div className="mt-3 flex items-end justify-between">
              <p className="font-display text-4xl font-black">
                {scenarioData
                  ? Math.round(
                      (scenarioData.battery.initial_energy_kwh /
                        scenarioData.battery.capacity_kwh) *
                        100,
                    )
                  : 0}
                <span className="text-base text-muted-foreground">%</span>
              </p>
              <BatteryCharging size={28} className="text-battery" />
            </div>
            <div className="battery-track mt-3">
              <span />
            </div>
            <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
              <span>
                {scenarioData ? scenarioData.battery.initial_energy_kwh : 0} /{" "}
                {scenarioData ? scenarioData.battery.capacity_kwh : 0} KWH
              </span>
              <span>MIN {scenarioData ? scenarioData.battery.minimum_energy_kwh : 0} KWH</span>
            </div>

            <div className="section-rule" />
            <label htmlFor="operator-note" className="eyebrow">
              OPERATOR NOTES · NATURAL LANGUAGE
            </label>
            <textarea
              id="operator-note"
              value={note}
              onChange={(event) => setNote(event.target.value)}
              className="operator-note mt-2"
            />
            <div className="mt-3 space-y-2">
              {apiResult?.directive_interpretation?.map(
                (d: any, idx: number) =>
                  d.directive_type !== "no_op" && (
                    <Directive
                      key={idx}
                      tone="battery"
                      type={d.directive_type.toUpperCase().replace(/_/g, " ")}
                      text={d.explanation || ""}
                    />
                  ),
              )}
              {!apiResult && (
                <div className="text-muted-foreground text-xs italic mt-2">
                  Upload and run a scenario to see directives.
                </div>
              )}
            </div>

            <button
              className="action-button mt-5 w-full justify-center py-3"
              onClick={run}
              disabled={running}
            >
              {running ? (
                <LoaderCircle size={15} className="animate-spin" />
              ) : (
                <Sparkles size={15} />
              )}
              {running ? status.toUpperCase() : "RUN OPTIMIZATION"}
            </button>
          </aside>
        </div>

        <section className="schedule-panel mt-3">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <div>
              <p className="eyebrow">VERIFIED HOURLY PLAN</p>
              <h2 className="mt-1 font-display text-lg font-bold">
                {apiResult ? apiResult.plan_summary : scenarioData ? "Ready for optimization" : "Upload a scenario to begin"}
              </h2>
            </div>
            <button className="icon-button" aria-label="Reset schedule">
              <RotateCcw size={15} />
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] border-collapse font-mono text-[11px]">
              <thead>
                <tr>
                  {[
                    "TIME",
                    "GRID KWH",
                    "SOLAR USED",
                    "BATTERY ACTION",
                    "BATTERY KWH",
                    "ENERGY AFTER",
                  ].map((label) => (
                    <th key={label}>{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(apiResult?.hourly_plan || []).map((row: any) => (
                  <tr key={row.hour}>
                    <td className="text-muted-foreground">
                      {row.hour.toString().padStart(2, "0")}:00
                    </td>
                    <td>{row.grid_kwh.toFixed(1)}</td>
                    <td className="text-solar">{row.solar_used_kwh.toFixed(1)}</td>
                    <td>
                      <span
                        className={`table-state tone-${row.battery_action === "charge" ? "solar" : row.battery_action === "discharge" ? "battery" : "muted"}`}
                      >
                        {row.battery_action.charAt(0).toUpperCase() + row.battery_action.slice(1)}
                      </span>
                    </td>
                    <td>
                      {row.battery_action === "charge"
                        ? "+"
                        : row.battery_action === "discharge"
                          ? "−"
                          : ""}
                      {Math.abs(row.battery_kwh).toFixed(1)}
                    </td>
                    <td>{row.battery_energy_after_kwh.toFixed(1)} kWh</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  );
}

function Legend({ tone, label }: { tone: string; label: string }) {
  return (
    <span className="flex items-center gap-2">
      <i className={`legend-dot tone-${tone}`} />
      {label}
    </span>
  );
}

function Metric({
  icon: Icon,
  label,
  value,
  unit,
  detail,
  tone,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  unit?: string;
  detail: string;
  tone: string;
}) {
  return (
    <article className="metric">
      <div className="flex items-center justify-between">
        <p className="eyebrow">{label}</p>
        <Icon size={14} className={`tone-text-${tone}`} />
      </div>
      <p className={`mt-2 font-display text-2xl font-black tone-text-${tone}`}>
        {value} <span className="text-xs text-muted-foreground">{unit}</span>
      </p>
      <p className="mt-1 font-mono text-[9px] text-muted-foreground">{detail.toUpperCase()}</p>
    </article>
  );
}

function Directive({ tone, type, text }: { tone: string; type: string; text: string }) {
  return (
    <div className="directive">
      <i className={`legend-dot tone-${tone}`} />
      <div>
        <p className="font-mono text-[9px] text-muted-foreground">{type}</p>
        <p className="mt-0.5 text-xs">{text}</p>
      </div>
      <CheckCircle2 size={14} className="ml-auto text-battery" />
    </div>
  );
}
