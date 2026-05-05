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
    const yMax = Math.max(...train, ...val) * 1.1;
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

  return { drawSwissMap, drawLossCurve, drawForecast, sparkline, genSpark };
})();
