/* =====================================================================
   LIULIAN demo · custom SVG chart renderers
   - Switzerland map (data agent)
   - Loss curve (training)
   - Forecast chart with confidence band (insight)
   - Sparklines (KPIs)
   ===================================================================== */

const SVG_NS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

const Charts = (() => {

  /* ---- Switzerland map ----------------------------------------- */
  function drawSwissMap(host) {
    host.innerHTML = "";
    const svg = el("svg", {
      class: "swiss-map",
      viewBox: "0 0 500 320",
      preserveAspectRatio: "xMidYMid meet",
    }, host);

    /* Country outline — simplified abstract Swiss-like silhouette */
    el("path", {
      class: "country",
      d: "M 50 200 C 70 188, 100 180, 130 178 C 165 175, 195 178, 225 178 C 255 178, 285 180, 320 184 C 350 188, 385 192, 415 200 C 445 208, 465 218, 470 230 C 470 244, 458 258, 432 268 C 400 278, 360 285, 320 288 C 280 290, 240 290, 200 286 C 165 282, 130 275, 100 265 C 75 255, 55 240, 48 222 C 45 210, 47 205, 50 200 Z",
    }, svg);

    /* Decorative topographic lines */
    MOCK.topoLines.forEach((d) => {
      el("path", { class: "topo", d }, svg);
    });

    /* Rivers */
    MOCK.rivers.forEach((d, i) => {
      el("path", { class: `river r${i + 1}`, d }, svg);
    });

    /* Stations — outer ring + inner core for visibility */
    MOCK.stations.forEach((s, i) => {
      const delay = 0.6 + i * 0.04;
      const cls = s.active ? "station station-active" : "station";
      const r = s.active ? 5.5 : 3.8;
      // Outer halo ring for every station (always visible silhouette)
      el("circle", {
        cx: s.x, cy: s.y, r: r + 2.5,
        fill: "rgba(79, 251, 233, 0.10)",
        stroke: "rgba(79, 251, 233, 0.25)",
        "stroke-width": 0.6,
        style: `opacity:0; animation: stationPop 0.4s var(--ease-spring) ${delay}s forwards;`,
      }, svg);
      el("circle", {
        class: cls,
        cx: s.x, cy: s.y, r,
        style: `animation-delay: ${delay}s`,
      }, svg);
      if (s.active) {
        el("circle", {
          class: "station-pulse",
          cx: s.x, cy: s.y,
          r: 5,
          style: `animation-delay: ${delay + 0.2}s; stroke-width: 1.6;`,
        }, svg);
      }
    });

    /* Selected labels for context */
    [
      { x: 88,  y: 178, t: "Rhein-Basel"  },
      { x: 222, y: 260, t: "Aare-Bern"    },
      { x: 380, y: 175, t: "Rhein-Diepoldsau" },
    ].forEach(({ x, y, t }) => {
      el("text", { class: "label", x: x + 8, y: y - 6 }, svg).textContent = t;
    });

  }

  /* ---- Loss curve --------------------------------------------- */
  function drawLossCurve(host) {
    host.innerHTML = "";
    const W = 480, H = 200, PAD = { l: 40, r: 12, t: 14, b: 22 };
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: "none",
    }, host);

    /* Defs for gradient under curve — red, soft */
    const defs = el("defs", {}, svg);
    const grad = el("linearGradient", {
      id: "lossGrad",
      x1: "0", y1: "0", x2: "0", y2: "1",
    }, defs);
    el("stop", { offset: "0%", "stop-color": "#E20613", "stop-opacity": "0.10" }, grad);
    el("stop", { offset: "100%", "stop-color": "#E20613", "stop-opacity": "0" }, grad);

    const { train, val } = MOCK.lossCurve(80);
    const rawMax = Math.max(...train, ...val);
    // Round up to a tidy gridline (0.25 step) to avoid axis labels like 1.02
    const yMax = Math.ceil(rawMax * 4) / 4;
    const yMin = 0;

    const xScale = (i) => PAD.l + (W - PAD.l - PAD.r) * (i / (train.length - 1));
    const yScale = (v) => PAD.t + (H - PAD.t - PAD.b) * (1 - (v - yMin) / (yMax - yMin));

    /* Grid lines (horizontal) */
    [0.2, 0.4, 0.6, 0.8].forEach((f) => {
      const y = PAD.t + (H - PAD.t - PAD.b) * f;
      el("line", {
        class: "grid-line",
        x1: PAD.l, y1: y, x2: W - PAD.r, y2: y,
      }, svg);
    });

    /* Y-axis labels */
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const v = yMin + (yMax - yMin) * (1 - f);
      const y = PAD.t + (H - PAD.t - PAD.b) * f;
      el("text", {
        class: "axis-label",
        x: PAD.l - 6, y: y + 3,
        "text-anchor": "end",
      }, svg).textContent = v.toFixed(2);
    });

    /* X-axis labels */
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const x = PAD.l + (W - PAD.l - PAD.r) * f;
      el("text", {
        class: "axis-label",
        x, y: H - 6, "text-anchor": "middle",
      }, svg).textContent = `${Math.round(f * 80)}`;
    });

    /* Axes */
    el("line", {
      class: "axis",
      x1: PAD.l, y1: H - PAD.b, x2: W - PAD.r, y2: H - PAD.b,
    }, svg);
    el("line", {
      class: "axis",
      x1: PAD.l, y1: PAD.t, x2: PAD.l, y2: H - PAD.b,
    }, svg);

    /* Train area */
    const areaPts = train.map((v, i) => `${xScale(i)} ${yScale(v)}`).join(" L ");
    const areaD = `M ${PAD.l} ${H - PAD.b} L ${areaPts} L ${W - PAD.r} ${H - PAD.b} Z`;
    el("path", { class: "loss-area", d: areaD }, svg);

    /* Val line */
    const valD = "M " + val.map((v, i) => `${xScale(i)} ${yScale(v)}`).join(" L ");
    el("path", { class: "loss-line val", d: valD }, svg);

    /* Train line (drawn last so it sits on top) */
    const trainD = "M " + train.map((v, i) => `${xScale(i)} ${yScale(v)}`).join(" L ");
    el("path", { class: "loss-line", d: trainD }, svg);

    /* Final-point dot for train */
    const lastT = train[train.length - 1];
    el("circle", {
      cx: xScale(train.length - 1),
      cy: yScale(lastT),
      r: 3.5,
      fill: "#E20613",
      stroke: "#FFFFFF",
      "stroke-width": 1.5,
      style: "opacity:0; animation: fadeIn 0.4s ease 2.6s forwards;",
    }, svg);
  }

  /* ---- Forecast chart ----------------------------------------- */
  function drawForecast(host) {
    host.innerHTML = "";
    const W = 720, H = 240, PAD = { l: 44, r: 14, t: 14, b: 28 };
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: "none",
    }, host);

    const defs = el("defs", {}, svg);
    const grad = el("linearGradient", {
      id: "ciGrad",
      x1: "0", y1: "0", x2: "0", y2: "1",
    }, defs);
    el("stop", { offset: "0%", "stop-color": "#E20613", "stop-opacity": "0.22" }, grad);
    el("stop", { offset: "100%", "stop-color": "#E20613", "stop-opacity": "0.03" }, grad);

    const { actual, pred, lo, hi, horizon, total } = MOCK.forecast();
    const yMax = Math.max(...hi) * 1.06;
    const yMin = Math.min(...lo) * 0.94;
    const xScale = (i) => PAD.l + (W - PAD.l - PAD.r) * (i / (total - 1));
    const yScale = (v) => PAD.t + (H - PAD.t - PAD.b) * (1 - (v - yMin) / (yMax - yMin));

    /* Grid */
    [0.25, 0.5, 0.75].forEach((f) => {
      const y = PAD.t + (H - PAD.t - PAD.b) * f;
      el("line", {
        class: "grid-line",
        x1: PAD.l, y1: y, x2: W - PAD.r, y2: y,
      }, svg);
    });

    /* Confidence band */
    let ciD = `M ${xScale(total - horizon)} ${yScale(hi[total - horizon])}`;
    for (let i = total - horizon; i < total; i++) {
      ciD += ` L ${xScale(i)} ${yScale(hi[i])}`;
    }
    for (let i = total - 1; i >= total - horizon; i--) {
      ciD += ` L ${xScale(i)} ${yScale(lo[i])}`;
    }
    ciD += " Z";
    el("path", { class: "ci-band", d: ciD }, svg);

    /* Actual line */
    const actualD = "M " + actual.map((v, i) => `${xScale(i)} ${yScale(v)}`).join(" L ");
    el("path", { class: "actual-line", d: actualD }, svg);

    /* Predicted line — over full range */
    const predD = "M " + pred.map((v, i) => `${xScale(i)} ${yScale(v)}`).join(" L ");
    el("path", { class: "pred-line", d: predD }, svg);

    /* Now line */
    const nowX = xScale(total - horizon);
    el("line", {
      class: "now-line",
      x1: nowX, y1: PAD.t, x2: nowX, y2: H - PAD.b,
    }, svg);
    el("text", {
      class: "now-label",
      x: nowX + 6, y: PAD.t + 12,
    }, svg).textContent = "NOW · 14:38 UTC";

    /* Threshold */
    const threshold = 0.85;
    const thrY = yScale(threshold);
    el("line", {
      class: "threshold",
      x1: PAD.l, y1: thrY, x2: W - PAD.r, y2: thrY,
    }, svg);
    el("text", {
      class: "label",
      x: W - PAD.r - 4, y: thrY - 4,
      "text-anchor": "end",
      fill: "#B00010",
    }, svg).textContent = "FLOOD THRESHOLD · 2 400 m³/s";

    /* Crossing marker — place at the exceeding point in horizon */
    let cIdx = total - 1;
    for (let i = total - horizon; i < total; i++) {
      if (pred[i] > threshold) { cIdx = i; break; }
    }
    el("circle", {
      class: "marker",
      cx: xScale(cIdx),
      cy: yScale(pred[cIdx]),
      r: 6,
    }, svg);
    el("text", {
      class: "marker-label",
      x: xScale(cIdx) + 10, y: yScale(pred[cIdx]) - 8,
    }, svg).textContent = "T+18h · CROSSED · P=0.87";

    /* Y-axis labels */
    [0, 0.5, 1].forEach((f) => {
      const v = yMin + (yMax - yMin) * (1 - f);
      const y = PAD.t + (H - PAD.t - PAD.b) * f;
      el("text", {
        class: "label",
        x: PAD.l - 6, y: y + 3, "text-anchor": "end",
      }, svg).textContent = `${(v * 3000).toFixed(0)}`;
    });
    el("text", {
      class: "label",
      x: PAD.l - 6, y: PAD.t - 6, "text-anchor": "end",
    }, svg).textContent = "m³/s";

    /* X-axis labels */
    const dayLabels = ["May 4", "May 5", "May 6", "May 7", "May 8", "May 9"];
    dayLabels.forEach((lab, i) => {
      const f = i / (dayLabels.length - 1);
      const x = PAD.l + (W - PAD.l - PAD.r) * f;
      el("text", {
        class: "label",
        x, y: H - 8, "text-anchor": "middle",
      }, svg).textContent = lab;
    });

    /* Axes */
    el("line", {
      class: "axis",
      x1: PAD.l, y1: H - PAD.b, x2: W - PAD.r, y2: H - PAD.b,
    }, svg);
  }

  /* ---- Sparkline ---------------------------------------------- */
  function sparkline(host, points, opts = {}) {
    host.innerHTML = "";
    const W = 60, H = 24;
    const svg = el("svg", {
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: "none",
    }, host);
    const max = Math.max(...points), min = Math.min(...points);
    const x = (i) => (W * i) / (points.length - 1);
    const y = (v) => H - 2 - (H - 4) * ((v - min) / (max - min || 1));
    const d = "M " + points.map((p, i) => `${x(i)} ${y(p)}`).join(" L ");
    el("path", {
      d,
      stroke: opts.color || "#E20613",
      "stroke-width": 1.4,
      fill: "none",
    }, svg);
    return svg;
  }

  function genSpark(seed = 0, n = 16) {
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push(0.5 + 0.4 * Math.sin(i * 0.6 + seed) + 0.15 * Math.cos(i * 1.3 + seed));
    }
    return out;
  }

  /* ---- Realistic Swiss map (used in Insight river-network section) -- */
  /* ViewBox 0 0 1000 580. Coordinates are illustrative — NOT survey-grade,
     but recognisable: Geneva SW, Basel N, Engadine SE, Ticino S tip. */
  const COUNTRY_PATH =
    "M 80 400 L 35 425 L 60 450 L 115 478 L 200 498 L 280 525 L 360 555 " +
    "L 430 568 L 480 555 L 510 540 L 535 545 L 560 565 L 595 595 L 615 600 " +
    "L 605 575 L 600 555 L 595 525 L 605 510 L 625 488 L 660 468 L 700 452 " +
    "L 745 438 L 790 412 L 825 370 L 870 340 L 920 320 L 960 305 L 980 280 " +
    "L 985 235 L 970 195 L 950 165 L 920 140 L 890 130 L 850 128 L 815 118 " +
    "L 780 100 L 750 75 L 700 60 L 645 50 L 600 35 L 560 22 L 540 38 " +
    "L 495 52 L 445 60 L 395 65 L 345 72 L 295 78 L 245 95 L 195 130 " +
    "L 165 175 L 145 220 L 130 280 L 110 330 L 95 365 Z";

  const RIVERS_REAL = [
    { name: "Rhein",   d: "M 870 130 L 820 110 L 780 102 L 750 90 L 720 78 L 685 65 L 640 58 L 600 38 L 560 22 L 540 38 L 495 52 L 445 60 L 395 65 L 345 72 L 295 78" },
    { name: "Rhône",   d: "M 410 480 L 360 488 L 310 495 L 270 482 L 230 460 L 195 440 L 165 425 L 130 415 L 95 405 L 60 395" },
    { name: "Aare",    d: "M 380 415 L 388 365 L 378 320 L 360 295 L 335 290 Q 305 290 280 305 L 260 320 L 250 345 L 270 370 L 295 360 L 325 335 L 360 285 L 390 230 L 420 180 L 440 145 L 445 110 L 430 88" },
    { name: "Reuss",   d: "M 555 360 L 540 310 L 525 280 L 520 240 L 510 200 L 480 165 L 455 140 L 440 120" },
    { name: "Limmat",  d: "M 590 145 L 555 130 L 510 130 L 470 130" },
    { name: "Inn",     d: "M 820 360 L 855 335 L 890 305 L 925 275 L 955 245" },
    { name: "Ticino",  d: "M 595 380 L 615 440 L 625 490 L 630 545 L 615 600" },
  ];

  const LAKES_REAL = [
    { name: "Léman",      cx: 155, cy: 430, rx: 90, ry: 14, rot:  8 },
    { name: "Constance",  cx: 725, cy:  80, rx: 80, ry: 14, rot: -5 },
    { name: "Neuchâtel",  cx: 220, cy: 245, rx: 38, ry: 11, rot: 35 },
    { name: "Zürich",     cx: 605, cy: 150, rx: 38, ry:  9, rot:-30 },
    { name: "Lucerne",    cx: 525, cy: 232, rx: 22, ry: 13, rot: 20 },
    { name: "Brienz",     cx: 405, cy: 295, rx: 20, ry:  6, rot:-12 },
    { name: "Thun",       cx: 360, cy: 285, rx: 22, ry:  6, rot:-15 },
    { name: "Maggiore",   cx: 615, cy: 555, rx: 25, ry: 35, rot: 10, partial: true },
  ];

  /* Real-station locations (approximate). The first two are highlighted. */
  const STATIONS_REAL = [
    { code: "RHE-BS", name: "Rhein-Basel",          x: 290, y:  78, active: true,  flagged: true  },
    { code: "AAR-BE", name: "Aare-Bern",            x: 268, y: 318, active: true,  upstream: true },
    { code: "AAR-BG", name: "Aare-Brugg",           x: 432, y: 122, upstream: true },
    { code: "LIM-BA", name: "Limmat-Baden",         x: 478, y: 134 },
    { code: "REU-LZ", name: "Reuss-Luzern",         x: 525, y: 240 },
    { code: "AAR-TH", name: "Aare-Thun",            x: 380, y: 295, upstream: true },
    { code: "RHO-SI", name: "Rhône-Sion",           x: 315, y: 510 },
    { code: "RHO-BR", name: "Rhône-Brig",           x: 390, y: 540 },
    { code: "RHE-DI", name: "Rhein-Diepoldsau",     x: 835, y: 145 },
    { code: "INN-SC", name: "Inn-S-chanf",          x: 885, y: 320 },
    { code: "TIC-BL", name: "Ticino-Bellinzona",    x: 645, y: 470 },
    { code: "LIN-ML", name: "Linth-Mollis",         x: 660, y: 215 },
    { code: "THU-AN", name: "Thur-Andelfingen",     x: 555, y:  85 },
    { code: "DOU-GM", name: "Doubs-Goumois",        x: 170, y: 130 },
    { code: "SAA-LP", name: "Saane-Laupen",         x: 260, y: 280 },
    { code: "RHE-DM", name: "Rhein-Domat",          x: 765, y: 240 },
    { code: "RHO-GE", name: "Rhône-Genève",         x:  78, y: 405 },
    { code: "SIH-ZH", name: "Sihl-Zürich",          x: 605, y: 130 },
    { code: "AAR-OL", name: "Aare-Olten",           x: 365, y: 165 },
    { code: "BOD-RO", name: "Bodensee-Romanshorn",  x: 760, y:  65 },
    { code: "REU-ML", name: "Reuss-Mellingen",      x: 490, y: 175 },
    { code: "MAG-LO", name: "Maggia-Locarno",       x: 595, y: 510 },
    { code: "MUR-FR", name: "Murg-Frauenfeld",      x: 605, y:  98 },
    { code: "EMM-EM", name: "Emme-Emmenmatt",       x: 320, y: 240 },
    { code: "JUR-LJ", name: "Jura-La Joux",         x: 145, y: 215 },
    { code: "BRO-PA", name: "Broye-Payerne",        x: 230, y: 290 },
    { code: "LAN-SC", name: "Landwasser-Schmitten", x: 800, y: 280 },
    { code: "VOR-FE", name: "Vorderrhein-Felsberg", x: 740, y: 215 },
    { code: "MOE-AL", name: "Moësa-Soazza",         x: 720, y: 365 },
    { code: "TIC-RI", name: "Tessin-Riazzino",      x: 635, y: 480 },
  ];

  function drawRealSwissMap(host, opts = {}) {
    if (!host) return;
    host.innerHTML = "";
    const annotate = opts.annotate !== false;

    const svg = el("svg", {
      class: "real-swiss-map",
      viewBox: "0 0 1000 620",
      preserveAspectRatio: "xMidYMid meet",
    }, host);

    /* Country fill */
    el("path", {
      class: "rs-country",
      d: COUNTRY_PATH,
    }, svg);

    /* Lakes */
    LAKES_REAL.forEach((lk) => {
      el("ellipse", {
        class: "rs-lake",
        cx: lk.cx, cy: lk.cy, rx: lk.rx, ry: lk.ry,
        transform: `rotate(${lk.rot} ${lk.cx} ${lk.cy})`,
      }, svg);
    });

    /* Rivers */
    RIVERS_REAL.forEach((r, i) => {
      el("path", {
        class: "rs-river",
        d: r.d,
        style: `animation-delay: ${0.3 + i * 0.12}s;`,
      }, svg);
    });

    /* Stations — base layer (gray dots) */
    STATIONS_REAL.forEach((s, i) => {
      if (s.flagged) return; // drawn last, on top
      el("circle", {
        class: s.upstream ? "rs-station rs-upstream" : "rs-station",
        cx: s.x, cy: s.y,
        r: s.upstream ? 4.5 : 3.2,
        style: `animation-delay: ${0.6 + i * 0.018}s;`,
      }, svg);
    });

    /* Active basin halo (faint red shading near flagged station) */
    if (annotate) {
      const flagged = STATIONS_REAL.find((s) => s.flagged);
      if (flagged) {
        el("circle", {
          class: "rs-halo",
          cx: flagged.x, cy: flagged.y,
          r: 38,
        }, svg);
        /* Outer ripple ring */
        el("circle", {
          class: "rs-ripple",
          cx: flagged.x, cy: flagged.y,
          r: 22,
        }, svg);
        /* Flagged station marker on top */
        el("circle", {
          class: "rs-station rs-flagged",
          cx: flagged.x, cy: flagged.y,
          r: 6.5,
        }, svg);
        /* Crossing-time annotation — placed to the right where there's room */
        el("line", {
          class: "rs-annot-line",
          x1: flagged.x + 8, y1: flagged.y + 2,
          x2: flagged.x + 110, y2: flagged.y + 30,
        }, svg);
        const tx = flagged.x + 120;
        const ty = flagged.y + 38;
        el("text", {
          class: "rs-annot-label",
          x: tx, y: ty - 16,
        }, svg).textContent = "RHEIN-BASEL · RHE-BS";
        el("text", {
          class: "rs-annot-value",
          x: tx, y: ty,
        }, svg).textContent = "T+18 h · 2 412 m³/s · p = 0.87";
        el("text", {
          class: "rs-annot-sub",
          x: tx, y: ty + 16,
        }, svg).textContent = "elevated · drainage 36 472 km²";
      }

      /* Upstream contributor labels */
      const upstreamLabels = [
        { code: "AAR-BE", offX: 14, offY: 16 },
        { code: "AAR-BG", offX: 14, offY: 16 },
        { code: "AAR-TH", offX: -8, offY: 22 },
      ];
      upstreamLabels.forEach(({ code, offX, offY }) => {
        const s = STATIONS_REAL.find((x) => x.code === code);
        if (!s) return;
        el("text", {
          class: "rs-up-label",
          x: s.x + offX,
          y: s.y + offY,
        }, svg).textContent = code;
      });
    }

    /* Cardinal labels */
    el("text", { class: "rs-card", x:  60, y:  18 }, svg).textContent = "N";
    el("text", { class: "rs-card", x: 988, y: 318 }, svg).textContent = "E";
    el("text", { class: "rs-card", x: 980, y: 612 }, svg).textContent = "S";
    el("text", { class: "rs-card", x:  10, y: 412 }, svg).textContent = "W";

    /* Major-city labels (helps establish geography) */
    [
      { name: "GENÈVE",    x:  68, y: 408 },
      { name: "BERN",      x: 252, y: 318, brand: true },
      { name: "BASEL",     x: 268, y:  72, brand: true },
      { name: "ZÜRICH",    x: 588, y: 132 },
      { name: "LUGANO",    x: 595, y: 555 },
      { name: "ST GALLEN", x: 758, y: 132 },
      { name: "CHUR",      x: 762, y: 230 },
    ].forEach(({ name, x, y, brand }) => {
      el("text", {
        class: "rs-city" + (brand ? " brand" : ""),
        x, y,
      }, svg).textContent = name;
    });
  }

  return {
    drawSwissMap, drawLossCurve, drawForecast, sparkline, genSpark,
    drawRealSwissMap,
    STATIONS_REAL,
  };
})();
