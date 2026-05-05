/* =====================================================================
   LIULIAN demo · mock data generators
   All numbers are illustrative — no real telemetry is involved.
   ===================================================================== */

const MOCK = (() => {

  /* Switzerland — stylised station coordinates inside the SVG viewBox.
     Coordinates picked so that the dot cluster reads as "Swiss-shaped"
     without claiming literal cartographic accuracy.                     */
  const stations = [
    { x: 88,  y: 178, name: "Rhein-Basel",     active: true,  flow: 1342 },
    { x: 165, y: 162, name: "Aare-Brugg",      active: false, flow: 318 },
    { x: 245, y: 158, name: "Limmat-Baden",    active: false, flow: 207 },
    { x: 312, y: 165, name: "Thur-Andelfingen",active: false, flow: 166 },
    { x: 380, y: 175, name: "Rhein-Diepoldsau",active: false, flow: 412 },
    { x: 445, y: 192, name: "Inn-S-chanf",     active: false, flow:  89 },
    { x: 200, y: 210, name: "Reuss-Luzern",    active: false, flow: 271 },
    { x: 280, y: 232, name: "Linth-Mollis",    active: false, flow: 144 },
    { x: 110, y: 240, name: "Rhône-Brig",      active: false, flow: 198 },
    { x: 168, y: 268, name: "Rhône-Sion",      active: false, flow: 232 },
    { x: 245, y: 285, name: "Ticino-Bellinzona", active: false, flow: 156 },
    { x: 64,  y: 232, name: "Doubs-Goumois",   active: false, flow:  72 },
    { x: 142, y: 218, name: "Saane-Laupen",    active: false, flow: 119 },
    { x: 222, y: 260, name: "Aare-Bern",       active: true,  flow: 506 },
    { x: 365, y: 220, name: "Rhein-Domat",     active: false, flow: 346 },
  ];

  /* River paths inside Switzerland (highly simplified) */
  const rivers = [
    "M 60 220 C 110 215, 150 205, 200 200 S 300 195, 360 200 S 420 200, 460 195",
    "M 95 178 C 140 172, 180 168, 230 175 S 320 195, 380 215 S 430 235, 460 248",
    "M 230 280 C 240 250, 250 220, 268 200 S 290 180, 312 165",
    "M 110 240 C 140 250, 170 270, 220 268 S 280 245, 320 230",
  ];

  /* Topographic decorative paths */
  const topoLines = [
    "M 30 195 C 80 175, 160 165, 250 158 S 380 165, 470 175",
    "M 35 215 C 100 195, 200 185, 290 192 S 420 200, 480 210",
    "M 45 240 C 120 230, 220 232, 310 240 S 420 250, 475 245",
    "M 50 270 C 130 268, 240 280, 330 282 S 425 275, 472 268",
  ];

  /* Streaming live-data rows */
  const streamSamples = [
    ["14:38:02", "AAR-BE", "548.2 m³/s", "norm"],
    ["14:38:01", "RHE-BS", "1342.7 m³/s", "norm"],
    ["14:38:01", "RHO-SI", "234.1 m³/s", "norm"],
    ["14:38:00", "LIN-ML", "144.8 m³/s", "norm"],
    ["14:37:59", "RHE-DI", "412.6 m³/s", "rise"],
    ["14:37:58", "AAR-BG", "318.3 m³/s", "norm"],
    ["14:37:57", "INN-SC", "  89.1 m³/s", "norm"],
    ["14:37:56", "TIC-BL", "156.4 m³/s", "norm"],
    ["14:37:55", "REU-LZ", "271.9 m³/s", "norm"],
    ["14:37:54", "DOU-GM", "  72.4 m³/s", "norm"],
    ["14:37:53", "SAA-LP", "119.5 m³/s", "norm"],
    ["14:37:52", "LIM-BA", "207.6 m³/s", "norm"],
    ["14:37:51", "THU-AN", "166.2 m³/s", "rise"],
    ["14:37:50", "RHO-BR", "198.7 m³/s", "norm"],
    ["14:37:49", "RHE-DM", "346.1 m³/s", "norm"],
  ];

  /* Loss curve (train + val) */
  function lossCurve(steps = 80) {
    const train = [], val = [];
    for (let i = 0; i < steps; i++) {
      const t = i / steps;
      const baseT = 0.78 * Math.exp(-i / 18) + 0.06;
      const baseV = 0.82 * Math.exp(-i / 22) + 0.082;
      const noiseT = (Math.sin(i * 0.7) + Math.cos(i * 1.4)) * 0.012;
      const noiseV = (Math.sin(i * 0.6 + 2) + Math.cos(i * 1.1 + 1)) * 0.018;
      train.push(Math.max(0.02, baseT + noiseT));
      val.push(Math.max(0.04, baseV + noiseV));
    }
    return { train, val };
  }

  /* HPO leaderboard rows */
  const hpoRows = [
    { rank: 1, model: "Transformer", config: "h=128 layers=4 lr=3e-4 ent=hash", score: 0.0418, delta: "−18.6%" },
    { rank: 2, model: "DLinear+E",   config: "h= 96 patch=24  lr=2e-4 ent=embed", score: 0.0432, delta: "−15.8%" },
    { rank: 3, model: "Mamba",       config: "h=128 d_state=16 lr=1e-3 ent=embed", score: 0.0451, delta: "−12.2%" },
    { rank: 4, model: "LSTM",        config: "h=128 layers=2 lr=1e-3 ent=onehot",  score: 0.0476, delta: " −7.5%" },
    { rank: 5, model: "ETSformer",   config: "h= 64 fourier=12 lr=2e-4 ent=hash",  score: 0.0492, delta: " −4.4%" },
    { rank: 6, model: "TSMixer",     config: "h= 96 mlp_ratio=4 lr=3e-4 ent=none", score: 0.0514, delta: " baseline" },
  ];

  /* Forecast (actual vs predicted) — 60 historical pts + 24 future */
  function forecast() {
    const total = 84, horizon = 24;
    const actual = [], pred = [], lo = [], hi = [];
    for (let i = 0; i < total; i++) {
      const trend = 0.55 + 0.18 * Math.sin(i / 12) + 0.06 * Math.sin(i / 4);
      const seasonal = 0.10 * Math.cos(i / 8 + 1) + 0.04 * Math.sin(i / 3);
      const noise = (Math.sin(i * 1.7) + Math.cos(i * 2.1)) * 0.025;
      const v = trend + seasonal + noise;
      if (i < total - horizon) actual.push(v);
      pred.push(i < total - horizon ? v + (Math.cos(i) * 0.012) : v);
      const w = i < total - horizon ? 0.018 : 0.04 + (i - (total - horizon)) * 0.008;
      lo.push(pred[i] - w);
      hi.push(pred[i] + w);
    }
    return { actual, pred, lo, hi, horizon, total };
  }

  /* KPI cards */
  const kpis = [
    { lab: "Stations Online", val: "2 143",  unit: "",   delta: "+12 today",     trend: "up"   },
    { lab: "Records Today",   val: "1.84",   unit: "M",  delta: "+8.4% vs MA7",  trend: "up"   },
    { lab: "Manifest Hash",   val: "OK",     unit: "",   delta: "sha256 verified", trend: "ok"   },
    { lab: "Latency p99",     val: "8.1",    unit: "ms", delta: "−1.4 ms vs SLA", trend: "up"   },
  ];

  /* Insight metrics */
  const insightCards = [
    { lab: "Anomalies (24h)",   val: "3", unit: "",       sub: "Bern-Rhône cluster" },
    { lab: "Risk Score",        val: "ELEVATED", unit: "", sub: "Threshold 2 400 m³/s",
      alert: true },
    { lab: "Forecast Accuracy", val: "94.2", unit: "%",   sub: "MAPE · 7-day rolling" },
    { lab: "Models in Pool",    val: "12", unit: "",      sub: "auto-selected ensemble" },
  ];

  /* Insight summary text — long-form, gets typed out */
  const insightText = [
    { t: "Discharge at " },
    { t: "Rhein-Basel", strong: true },
    { t: " is forecast to exceed " },
    { t: "2 400 m³/s", strong: true },
    { t: " on " },
    { t: "May 8 14:00 UTC", strong: true },
    { t: " with " },
    { t: "87 % confidence", strong: true },
    { t: ". Risk classification: " },
    { t: "ELEVATED", risk: true },
    { t: ". Recommend hydrology team review before 2026-05-07T18:00Z." },
  ];

  /* ---- v3 additions: per-screen content -------------------------- */

  /* Datasets list (data screen sidebar) */
  const datasets = [
    { id: "swiss-river-1990", name: "swiss-river-1990", span: "1990 – 2026", count: "2 143", selected: true },
    { id: "swiss-river-2000", name: "swiss-river-2000", span: "2000 – 2026", count: "1 892" },
    { id: "mch-radar-2020",   name: "mch-radar-2020",   span: "2020 – 2026", count: "5 × 5 km grid" },
    { id: "smn-meteo-1980",   name: "smn-meteo-1980",   span: "1980 – 2026", count: "468" },
    { id: "glaciers-vaw",     name: "glaciers-vaw",     span: "1959 – 2024", count: "21 sites" },
  ];

  /* Manifest YAML text (data screen, code block) */
  const manifest = [
    'name: swiss-river-1990',
    'schema: hydro-v2',
    'freq: 10min',
    'span: ["1990-01-01", "2026-05-05"]',
    'stations: 2143',
    'integrity:',
    '  algo: sha256',
    '  digest: a3b4c1f2…',
    'columns: [ts, station, value, qc, basin, lat, lon, elev]',
    'license: CC-BY-4.0',
  ];

  /* Schema columns (data screen sidebar) */
  const columns = [
    { name: "ts",       type: "datetime[UTC]", null: false },
    { name: "station",  type: "str",           null: false },
    { name: "value",    type: "float32",       null: true  },
    { name: "qc",       type: "uint8",         null: false },
    { name: "basin",    type: "str",           null: false },
    { name: "lat",      type: "float32",       null: false },
    { name: "lon",      type: "float32",       null: false },
    { name: "elev",     type: "int16",         null: true  },
  ];

  /* Data preview rows (data screen main) */
  const dataRows = [
    ["2026-05-05 14:38:00", "RHE-BS", "1342.7", "0", "Rhein"],
    ["2026-05-05 14:37:50", "AAR-BE", " 548.2", "0", "Aare"],
    ["2026-05-05 14:37:40", "RHE-DI", " 412.6", "1", "Rhein"],
    ["2026-05-05 14:37:30", "INN-SC", "  89.1", "0", "Inn"],
    ["2026-05-05 14:37:20", "REU-LZ", " 271.9", "0", "Reuss"],
    ["2026-05-05 14:37:10", "TIC-BL", " 156.4", "0", "Ticino"],
    ["2026-05-05 14:37:00", "LIM-BA", " 207.6", "0", "Limmat"],
    ["2026-05-05 14:36:50", "RHO-SI", " 234.1", "0", "Rhône"],
  ];

  /* Training runs table (train screen) */
  const runs = [
    { id: "t-002", model: "Transformer-E", status: "running", mse: 0.0418, epoch: "56/80", time: "14:23", best: true  },
    { id: "t-001", model: "DLinear-E",     status: "done",    mse: 0.0432, epoch: "80/80", time: "09:48" },
    { id: "t-000", model: "Mamba",         status: "done",    mse: 0.0451, epoch: "80/80", time: "12:11" },
    { id: "t-aaf", model: "LSTM",          status: "done",    mse: 0.0476, epoch: "80/80", time: "06:15" },
    { id: "t-93c", model: "ETSformer",     status: "done",    mse: 0.0492, epoch: "80/80", time: "11:02" },
    { id: "t-77b", model: "TSMixer",       status: "done",    mse: 0.0514, epoch: "80/80", time: "07:38" },
    { id: "t-55a", model: "DLinear",       status: "failed",  mse: null,   epoch: "32/80", time: "04:18" },
  ];

  /* Training config YAML (train screen, code block) */
  const trainConfig = [
    'model: transformer',
    'dim: 128',
    'layers: 4',
    'heads: 8',
    'entity_mode: hash',
    'lr: 3e-4',
    'batch: 64',
    'epochs: 80',
    'optimizer: adamw',
    'scheduler: cosine',
  ];

  /* Inference endpoint info */
  const endpoint = {
    method: "POST",
    url:    "api.liulian.ch/v1/forecast",
    desc:   "Forecast discharge at any Swiss station, 1–72 h horizon.",
    p50:    "2.3 ms",
    p99:    "8 ms",
    rps:    "1 240 req/s",
  };

  /* Inference code snippet (Python) */
  const codeSnippet = [
    'from liulian import client',
    '',
    'resp = client.forecast(',
    '    station="RHE-BS",',
    '    horizon=24,',
    '    ci=0.95,',
    ')',
    'print(resp.crossing)',
    '# 2026-05-08T14:00Z, p=0.87, elevated',
  ];

  /* Insight sessions (chat sidebar) */
  const sessions = [
    { id: "s-now",  title: "Flood risk this week",     when: "now",       active: true },
    { id: "s-001",  title: "Aare-Bern anomaly review", when: "yesterday" },
    { id: "s-002",  title: "Compare 2025 vs 2026",     when: "Apr 30" },
    { id: "s-003",  title: "Glacier melt scenario",    when: "Apr 24" },
    { id: "s-004",  title: "Sensor 412 outage",        when: "Apr 18" },
  ];

  /* Insight chat content */
  const chatMessages = [
    {
      role: "user",
      time: "14:38",
      text: "Is there flood risk in the next 7 days?",
    },
    {
      role: "agent",
      time: "14:38",
      author: "Insight",
      // segments either ASCII text or { strong, risk } markers — typed sequentially
      segments: [
        { t: "Yes. " },
        { t: "Rhein-Basel", strong: true },
        { t: " is forecast to exceed " },
        { t: "2 400 m³/s", strong: true },
        { t: " on " },
        { t: "May 8, ~14:00 UTC", strong: true },
        { t: ", with " },
        { t: "87 %", strong: true },
        { t: " confidence (" },
        { t: "elevated", risk: true },
        { t: ").\n\n" },
        { chart: true },
        { t: "\n\nThe crossing happens 18 hours from now. Confidence is tight (±60 m³/s); the upstream Aare contribution is well observed by Aare-Bern (AAR-BE) and Aare-Brugg (AAR-BG)." },
      ],
      sources: [
        { name: "swiss-river-1990",            kind: "data" },
        { name: "transformer-entity-aware-v3", kind: "model" },
        { name: "run t-002",                   kind: "run" },
        { name: "MCH precip 24h",              kind: "data" },
      ],
    },
  ];

  /* Insight context (right rail) */
  const context = {
    dataset: "swiss-river-1990",
    model:   "transformer-entity-aware-v3",
    runId:   "t-002",
    sources: 4,
    trail: [
      { label: "Fetch data",      done: true  },
      { label: "Run forecast",    done: true  },
      { label: "Reason over CI",  done: true  },
      { label: "Compose answer",  done: true  },
    ],
  };

  return {
    stations, rivers, topoLines, streamSamples,
    lossCurve, hpoRows, forecast, kpis, insightCards, insightText,
    /* v3 */
    datasets, manifest, columns, dataRows, runs, trainConfig,
    endpoint, codeSnippet, sessions, chatMessages, context,
  };
})();
