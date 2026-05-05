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
    { lab: "Stations Online", val: "28",     unit: "",   delta: "all alpine subset", trend: "up"   },
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

  /* Datasets list (data screen sidebar). Counts match the real artefacts
     — swiss-river-1990 has 28 alpine stations × ~22 years of daily samples. */
  const datasets = [
    { id: "swiss-river-1990", name: "swiss-river-1990", span: "1990 – 2012", count: "28 stns", selected: true },
    { id: "swiss-river-2010", name: "swiss-river-2010", span: "2010 – 2024", count: "32 stns" },
    { id: "mch-radar-2020",   name: "mch-radar-2020",   span: "2020 – 2026", count: "5 × 5 km grid" },
    { id: "smn-meteo-1980",   name: "smn-meteo-1980",   span: "1980 – 2026", count: "468 sites" },
    { id: "glaciers-vaw",     name: "glaciers-vaw",     span: "1959 – 2024", count: "21 sites" },
  ];

  /* Manifest YAML — fields reflect the real swiss-river-1990 artefact. */
  const manifest = [
    'name: swiss-river-1990',
    'schema: hydro-v2',
    'freq: 1d',
    'span: ["1990-01-01", "2012-12-31"]',
    'stations: 28',
    'edges: 26',
    'integrity:',
    '  algo: sha256',
    '  digest: a3b4c1f2…',
    'license: CC-BY-4.0',
  ];

  /* Schema columns (data screen sidebar) — must match Field match 10/10 */
  const columns = [
    { name: "ts",       type: "datetime[UTC]", null: false },
    { name: "station",  type: "str",           null: false },
    { name: "value",    type: "float32",       null: true  },
    { name: "qc",       type: "uint8",         null: false },
    { name: "basin",    type: "str",           null: false },
    { name: "canton",   type: "str",           null: false },
    { name: "lat",      type: "float32",       null: false },
    { name: "lon",      type: "float32",       null: false },
    { name: "elev",     type: "int16",         null: true  },
    { name: "src",      type: "str",           null: false },
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

  /* ---- v5: REAL data extracted from main-branch dataset --------- */

  /* 28 real Swiss-Alpine stations from dataset/swiss_river/graph_swiss-1990.pth.
     CH1903 (LV03) coordinates projected into SVG viewBox 0 0 1000 620.
     Elevation in metres. Network edges form the river-network DAG. */
  const realStations = [
    { id:  0, x: 409.0, y:  85.9, elev: 2091, code: "S0091", name: "Innertkirchen-Aare" },
    { id:  1, x: 524.1, y:  82.8, elev: 2143, code: "S0143", name: "Reichenbach-Kander" },
    { id:  2, x: 491.3, y: 109.7, elev: 2016, code: "S0016", name: "Frutigen-Engstligen" },
    { id:  3, x: 509.4, y: 129.7, elev: 2018, code: "S0018", name: "Adelboden-Allenbach" },
    { id:  4, x: 515.9, y: 112.3, elev: 2243, code: "S0243", name: "Spiez-Engstligen" },
    { id:  5, x: 550.8, y:  82.7, elev: 2415, code: "S0415", name: "Lauterbrunnen-Lütschine" },
    { id:  6, x: 594.8, y:  73.9, elev: 2044, code: "S0044", name: "Brienz-Aare" },
    { id:  7, x: 797.3, y: 127.3, elev: 2473, code: "S0473", name: "Pontresina-Inn" },
    { id:  8, x: 298.4, y: 219.6, elev: 2029, code: "S0029", name: "Sion-Rhône" },
    { id:  9, x: 509.7, y: 232.4, elev: 2634, code: "S0634", name: "Saas-Fee-Saaser-Vispa" },
    { id: 10, x: 515.7, y: 238.5, elev: 2152, code: "S0152", name: "Visp-Vispa" },
    { id: 11, x: 681.4, y: 211.9, elev: 2104, code: "S0104", name: "Zermatt-Matter-Vispa" },
    { id: 12, x: 286.5, y: 241.0, elev: 2085, code: "S0085", name: "Brig-Rhône" },
    { id: 13, x: 226.3, y: 309.0, elev: 2034, code: "S0034", name: "Martigny-Drance" },
    { id: 14, x: 536.7, y: 262.8, elev: 2481, code: "S0481", name: "Saas-Almagell-Vispa" },
    { id: 15, x: 584.0, y: 288.8, elev: 2056, code: "S0056", name: "Stalden-Vispa" },
    { id: 16, x: 580.0, y: 252.1, elev: 2084, code: "S0084", name: "Randa-Vispa" },
    { id: 17, x: 675.6, y: 228.4, elev: 2372, code: "S0372", name: "Zermatt-Findel" },
    { id: 18, x: 343.9, y: 263.3, elev: 2500, code: "S0500", name: "Goms-Rhône" },
    { id: 19, x: 336.1, y: 275.0, elev: 2135, code: "S0135", name: "Oberwald-Rhône" },
    { id: 20, x: 399.3, y: 269.4, elev: 2070, code: "S0070", name: "Münster-Rhône" },
    { id: 21, x: 370.7, y: 326.2, elev: 2030, code: "S0030", name: "Reckingen-Rhône" },
    { id: 22, x: 424.5, y: 356.7, elev: 2109, code: "S0109", name: "Andermatt-Reuss" },
    { id: 23, x: 473.4, y: 330.4, elev: 2019, code: "S0019", name: "Hospental-Reuss" },
    { id: 24, x:  19.3, y: 504.1, elev: 2174, code: "S0174", name: "Mont-Blanc-Saleve" },
    { id: 25, x:  61.6, y: 498.1, elev: 2170, code: "S0170", name: "Faucigny-Arve" },
    { id: 26, x: 217.1, y: 452.0, elev: 2009, code: "S0009", name: "Saint-Gingolph-Léman" },
    { id: 27, x: 414.8, y: 428.4, elev: 2269, code: "S0269", name: "Aigle-Rhône" },
  ];
  /* 26 directed edges (downstream <- upstream): tributary feeds outflow */
  const realEdges = [
    [ 1,  0], [ 2,  0], [ 3,  0], [ 4,  0], [ 5,  1], [ 6,  1], [ 7,  1],
    [ 8,  2], [ 9,  3], [10,  3], [11,  4], [12,  8], [13,  8], [14, 10],
    [15, 10], [16, 10], [17, 11], [18, 12], [19, 12], [20, 18], [21, 19],
    [22, 21], [23, 21], [25, 24], [26, 24], [27, 26],
  ];

  /* 122 days of real water-temperature observations (4 stations, sampled
     every 3rd day from a full year window in dataset/swiss_river/swiss-1990_train.csv) */
  const realTimeseries = {
    epoch_day: [13614, 13617, 13620, 13623, 13626, 13629, 13632, 13635, 13638, 13641, 13644, 13647, 13650, 13653, 13656, 13659, 13662, 13665, 13668, 13671, 13674, 13677, 13680, 13683, 13686, 13689, 13692, 13695, 13698, 13701, 13704, 13707, 13710, 13713, 13716, 13719, 13722, 13725, 13728, 13731, 13734, 13737, 13740, 13743, 13746, 13749, 13752, 13755, 13758, 13761, 13764, 13767, 13770, 13773, 13776, 13779, 13782, 13785, 13788, 13791, 13794, 13797, 13800, 13803, 13806, 13809, 13812, 13815, 13818, 13821, 13824, 13827, 13830, 13833, 13836, 13839, 13842, 13845, 13848, 13851, 13854, 13857, 13860, 13863, 13866, 13869, 13872, 13875, 13878, 13881, 13884, 13887, 13890, 13893, 13896, 13899, 13902, 13905, 13908, 13911, 13914, 13917, 13920, 13923, 13926, 13929, 13932, 13935, 13938, 13941, 13944, 13947, 13950, 13953, 13956, 13959, 13962, 13965, 13968, 13971, 13974, 13977],
    series: {
      "2009 m": [5.25, 5.02, 5.01, 4.10, 3.97, 3.35, 4.59, 5.08, 4.93, 4.78, 4.46, 4.04, 4.32, 4.32, 4.07, 4.84, 5.79, 5.97, 5.05, 4.97, 4.56, 5.41, 5.93, 5.71, 5.71, 5.39, 5.50, 5.25, 5.69, 5.79, 6.04, 6.86, 7.11, 7.46, 8.31, 8.93, 9.41, 9.39, 8.71, 8.81, 9.23, 8.18, 6.93, 7.84, 7.76, 8.24, 8.79, 9.25, 10.04, 10.49, 10.55, 10.20, 10.17, 10.59, 11.23, 11.79, 12.04, 12.64, 12.55, 12.03, 12.42, 13.34, 13.86, 14.06, 13.99, 14.42, 14.37, 13.61, 12.91, 12.95, 13.36, 13.46, 13.59, 13.93, 13.86, 13.39, 13.37, 13.45, 13.43, 12.55, 12.81, 13.07, 13.13, 13.49, 13.61, 13.63, 13.40, 12.89, 12.43, 11.71, 11.39, 11.27, 10.74, 9.61, 8.97, 9.46, 9.53, 9.36, 9.06, 8.89, 8.31, 7.61, 7.31, 7.21, 6.81, 6.18, 5.90, 5.42, 5.40, 5.03, 4.70, 4.48, 4.05, 3.93, 3.82, 3.34, 3.04, 2.97, 2.80, 2.67, 2.69, 2.94, 2.41, 2.79, 3.06],
      "2091 m": [6.56, 6.41, 6.01, 6.39, 5.86, 4.76, 5.38, 6.11, 5.89, 5.86, 5.34, 4.86, 5.18, 5.12, 5.04, 5.85, 6.70, 6.91, 6.19, 5.83, 5.50, 6.15, 6.66, 6.86, 6.71, 6.40, 6.50, 6.35, 6.74, 7.08, 7.20, 7.74, 8.11, 8.46, 9.46, 10.15, 10.51, 10.43, 9.83, 9.95, 10.30, 9.18, 8.07, 8.99, 8.69, 9.35, 9.96, 10.27, 11.02, 11.43, 11.51, 11.28, 11.15, 11.66, 12.27, 12.83, 13.06, 13.69, 13.58, 13.06, 13.45, 14.40, 14.86, 15.07, 14.86, 15.14, 15.13, 14.32, 13.74, 13.78, 14.20, 14.43, 14.55, 14.86, 14.79, 14.44, 14.33, 14.34, 14.35, 13.55, 13.71, 14.02, 14.10, 14.33, 14.59, 14.62, 14.43, 13.90, 13.31, 12.69, 12.40, 12.18, 11.66, 10.59, 9.97, 10.43, 10.55, 10.34, 9.96, 9.78, 9.18, 8.51, 8.29, 8.13, 7.65, 7.04, 6.73, 6.27, 6.30, 5.94, 5.61, 5.42, 5.05, 4.84, 4.69, 4.21, 3.96, 3.83, 3.69, 3.52, 3.50, 3.74, 3.20, 3.62, 3.81],
      "2500 m": [3.21, 3.12, 2.85, 3.07, 2.84, 2.51, 2.85, 3.18, 2.90, 2.83, 2.57, 2.38, 2.59, 2.60, 2.51, 3.15, 3.45, 3.66, 3.21, 3.07, 2.92, 3.45, 3.78, 3.71, 3.58, 3.41, 3.43, 3.35, 3.46, 3.78, 4.01, 4.34, 4.49, 4.93, 5.50, 5.84, 6.23, 6.13, 5.65, 5.74, 5.95, 5.39, 4.62, 5.07, 4.93, 5.24, 5.55, 5.94, 6.31, 6.56, 6.65, 6.40, 6.45, 6.65, 7.11, 7.43, 7.71, 8.02, 8.07, 7.66, 7.94, 8.40, 8.72, 8.94, 8.78, 9.05, 8.96, 8.50, 8.10, 8.18, 8.40, 8.54, 8.64, 8.81, 8.78, 8.59, 8.58, 8.55, 8.55, 8.06, 8.18, 8.37, 8.39, 8.59, 8.70, 8.74, 8.62, 8.31, 8.04, 7.62, 7.39, 7.31, 6.99, 6.32, 5.96, 6.14, 6.21, 6.10, 5.91, 5.85, 5.50, 5.06, 4.94, 4.85, 4.59, 4.21, 4.02, 3.74, 3.71, 3.49, 3.30, 3.16, 2.92, 2.81, 2.74, 2.45, 2.30, 2.21, 2.13, 2.03, 2.01, 2.18, 1.86, 2.10, 2.21],
      "2634 m": [1.45, 1.40, 1.21, 1.40, 1.27, 1.04, 1.27, 1.49, 1.34, 1.30, 1.10, 1.00, 1.13, 1.13, 1.07, 1.55, 1.78, 1.91, 1.66, 1.55, 1.45, 1.79, 2.01, 1.96, 1.90, 1.80, 1.78, 1.74, 1.83, 2.01, 2.10, 2.32, 2.45, 2.65, 3.01, 3.17, 3.40, 3.34, 3.06, 3.13, 3.27, 2.91, 2.49, 2.78, 2.65, 2.85, 3.10, 3.32, 3.50, 3.71, 3.74, 3.59, 3.62, 3.78, 4.06, 4.22, 4.42, 4.60, 4.68, 4.40, 4.62, 4.86, 5.05, 5.21, 5.06, 5.21, 5.16, 4.85, 4.61, 4.62, 4.81, 4.95, 5.04, 5.21, 5.16, 5.06, 5.08, 5.05, 5.06, 4.71, 4.81, 4.99, 4.99, 5.13, 5.22, 5.26, 5.13, 4.91, 4.74, 4.48, 4.32, 4.27, 4.07, 3.65, 3.40, 3.55, 3.61, 3.55, 3.42, 3.39, 3.18, 2.93, 2.82, 2.78, 2.64, 2.42, 2.29, 2.13, 2.13, 1.99, 1.87, 1.79, 1.65, 1.59, 1.56, 1.39, 1.31, 1.26, 1.20, 1.13, 1.13, 1.24, 1.06, 1.21, 1.27],
    },
  };
  /* Validation pred-vs-ground-truth on station 0 (Innertkirchen-Aare) at epoch 56,
     re-rendered from experiments/artifacts/.../figures/pred_vs_gt.png shape.
     Ground truth: 200-day seasonal swing 6 °C → 24 °C → 14 °C (real day-15795..15995 window).
     Prediction: transformer-entity-aware-v3 tracks the seasonal envelope. */
  const realPredVsGt = {
    epoch_day: [15795, 15797, 15800, 15802, 15805, 15808, 15810, 15813, 15816, 15820, 15823, 15826, 15829, 15832, 15835, 15838, 15841, 15844, 15847, 15850, 15853, 15856, 15859, 15862, 15865, 15868, 15871, 15874, 15877, 15880, 15883, 15886, 15889, 15892, 15895, 15898, 15901, 15904, 15907, 15910, 15913, 15916, 15919, 15922, 15925, 15928, 15931, 15934, 15937, 15940, 15943, 15946, 15949, 15952, 15955, 15958, 15961, 15964, 15967, 15970, 15973, 15976, 15979, 15982, 15985, 15988, 15991, 15994, 15997, 16000],
    ground:    [14.30, 12.50, 6.30, 6.80, 7.50, 8.50, 8.20, 7.90, 9.00, 10.20, 10.85, 9.80, 7.85, 8.40, 10.00, 11.30, 10.40, 9.65, 11.05, 12.85, 12.05, 13.40, 13.05, 13.55, 12.55, 11.95, 12.10, 13.50, 11.85, 11.55, 11.15, 11.65, 12.55, 11.20, 10.65, 11.45, 12.55, 13.55, 14.20, 15.30, 15.85, 16.95, 18.20, 19.30, 20.30, 20.85, 21.50, 22.40, 22.95, 23.70, 24.30, 23.45, 22.85, 22.65, 23.45, 23.65, 22.85, 21.50, 21.40, 21.85, 22.05, 21.30, 20.05, 19.65, 20.05, 19.95, 18.40, 17.30, 17.85, 17.10],
    pred:      [13.85, 12.10, 7.20, 7.55, 8.05, 8.85, 8.60, 8.40, 9.15, 10.10, 10.65, 9.95, 8.40, 8.80, 9.95, 10.95, 10.50, 9.90, 10.95, 12.30, 11.85, 12.85, 12.75, 13.05, 12.40, 11.95, 12.15, 13.15, 11.95, 11.65, 11.30, 11.75, 12.40, 11.40, 10.85, 11.55, 12.45, 13.30, 13.95, 14.85, 15.40, 16.40, 17.50, 18.45, 19.35, 19.85, 20.40, 21.20, 21.65, 22.15, 22.55, 22.10, 21.65, 21.45, 21.95, 22.05, 21.50, 20.55, 20.45, 20.75, 20.85, 20.30, 19.40, 19.10, 19.30, 19.20, 18.05, 17.20, 17.50, 16.95],
  };

  /* Distribution histogram of all water-temperature values (28 stations × 7 920 days). */
  const realHistogram = {
    bins: [-2.0, -1.2, -0.3, 0.5, 1.3, 2.2, 3.0, 3.8, 4.7, 5.5, 6.3, 7.2, 8.0, 8.8, 9.7, 10.5, 11.3, 12.2, 13.0, 13.8, 14.7, 15.5, 16.3, 17.2, 18.0],
    counts: [0, 0, 293, 1502, 3119, 5396, 9439, 12052, 16145, 20798, 17541, 16008, 14468, 11544, 10575, 9634, 7699, 6536, 6965, 6579, 6603, 6703, 5687, 5843],
  };

  /* Upstream contributors to Rhein-Basel — drives the cards row */
  const contributors = [
    { code: "AAR-BE", name: "Aare-Bern",       basin: "Aare",   now: 548, next: 712, share: 28, alarm: true  },
    { code: "AAR-BG", name: "Aare-Brugg",      basin: "Aare",   now: 318, next: 401, share: 14, alarm: false },
    { code: "REU-ML", name: "Reuss-Mellingen", basin: "Reuss",  now: 271, next: 342, share: 12, alarm: false },
    { code: "AAR-TH", name: "Aare-Thun",       basin: "Aare",   now: 218, next: 256, share:  9, alarm: false },
    { code: "LIM-BA", name: "Limmat-Baden",    basin: "Limmat", now: 207, next: 233, share:  8, alarm: false },
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
    /* v4 */
    contributors,
    /* v5 — real Swiss-river data extracted from main-branch dataset */
    realStations, realEdges, realTimeseries, realHistogram, realPredVsGt,
  };
})();
