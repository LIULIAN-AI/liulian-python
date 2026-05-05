/* =====================================================================
   LIULIAN demo · 20-second scene choreography
   Stages
     0.0–2.6 s  Hero
     2.6–3.0 s  Hero leaves
     3.0 s+     Dashboard reveal in waves
     14.0 s     Insight typing complete
     15.5 s     Production-ready badge
     17.0 s     Marker emphasis pulse
     18.0–20 s  Hold; loop on `?loop=1`.
   ===================================================================== */

const Scene = (() => {

  const params = new URLSearchParams(location.search);
  const speed = parseFloat(params.get("speed") || "1") || 1;
  const freeze = params.get("freeze");          // "hero" | "data" | "train" | "insight" | "finale"
  const stops = { hero: 1.5, data: 4.5, train: 7.5, insight: 13.5, finale: 19.0 };
  const freezeAt = freeze && stops[freeze] !== undefined ? stops[freeze] : null;

  const T = (t, fn) => {
    if (freezeAt !== null && t > freezeAt) return; // skip events after freeze point
    setTimeout(fn, (t * 1000) / speed);
  };

  let typingHandle = null;

  function buildHero() {
    const wm = document.querySelector(".hero-wordmark");
    if (!wm) return;
    const text = "LIULIAN";
    wm.innerHTML = "";
    [...text].forEach((ch, i) => {
      const span = document.createElement("span");
      if (i === 2) span.classList.add("accent"); // highlight U — the unique liquid character
      span.textContent = ch;
      wm.appendChild(span);
    });
  }

  function reveal(selector, delay = 0) {
    T(delay, () => {
      document.querySelectorAll(selector).forEach((c) => c.classList.add("is-visible"));
    });
  }

  function streamPopulate() {
    const host = document.querySelector(".stream-table");
    if (!host) return;
    const samples = MOCK.streamSamples;
    const max = 9; // visible rows
    let pointer = 0;

    function pushRow() {
      const s = samples[pointer % samples.length];
      pointer++;
      const row = document.createElement("div");
      row.className = "stream-row in";
      const tagCls = s[3] === "rise" ? "tag warn" : "tag";
      row.innerHTML = `
        <span class="ts">${s[0]}</span>
        <span class="id">${s[1]}</span>
        <span class="v">${s[2]}</span>
        <span class="${tagCls}">${s[3].toUpperCase()}</span>`;
      host.appendChild(row);
      if (host.children.length > max) host.removeChild(host.children[0]);
    }

    let count = 0;
    const start = 3.4; // seconds
    const interval = 0.34; // seconds between rows
    for (let i = 0; i < max + 4; i++) {
      T(start + i * interval, pushRow);
      count++;
    }
    // Continue feeding subtly during hold
    T(11.0, () => {
      const id = setInterval(pushRow, 800);
      // stop gracefully near the end
      setTimeout(() => clearInterval(id), 8000);
    });
  }

  function hpoPopulate() {
    const list = document.querySelector(".hpo-list");
    if (!list) return;
    list.innerHTML = "";
    const start = 6.4;
    MOCK.hpoRows.forEach((r, i) => {
      T(start + i * 0.18, () => {
        const div = document.createElement("div");
        div.className = "hpo-row in" + (i === 0 ? " top" : "");
        const cfg = r.config.replace(/(\w+)=/g, '<span class="key">$1=</span>');
        div.innerHTML = `
          <span class="rank">${r.rank}</span>
          <span class="config"><strong>${r.model}</strong>  ${cfg}</span>
          <span class="score">${r.score.toFixed(4)}</span>
          <span class="delta">${r.delta}</span>`;
        list.appendChild(div);
      });
    });
  }

  function kpiPopulate() {
    const grid = document.querySelector(".kpi-grid");
    if (!grid) return;
    grid.innerHTML = "";
    MOCK.kpis.forEach((k, i) => {
      const card = document.createElement("div");
      card.className = "kpi";
      const deltaCls = k.trend === "down" ? "delta down" : "delta";
      card.innerHTML = `
        <div class="lab">${k.lab}</div>
        <div class="val">${k.val}<span class="unit">${k.unit}</span></div>
        <div class="${deltaCls}">${k.delta}</div>
        <div class="spark"></div>`;
      grid.appendChild(card);
      Charts.sparkline(
        card.querySelector(".spark"),
        Charts.genSpark(i + 1, 18),
        { color: i === 2 ? "#346538" : "#E20613" },
      );
    });
  }

  function insightCardsPopulate() {
    const host = document.querySelector(".insight-cards");
    if (!host) return;
    host.innerHTML = "";
    MOCK.insightCards.forEach((c) => {
      const card = document.createElement("div");
      card.className = "ins-card" + (c.alert ? " alert" : "");
      card.innerHTML = `
        <div class="lab">${c.lab}</div>
        <div class="val">${c.val}<span class="unit">${c.unit}</span></div>
        <div class="sub">${c.sub}</div>`;
      host.appendChild(card);
    });
  }

  /* Type out the insight summary. */
  function typeInsight() {
    const host = document.querySelector(".insight-summary");
    if (!host) return;
    host.innerHTML = '<span class="agent-tag">INSIGHT AGENT · gpt-fc-7</span><span class="content"></span><span class="typed-cursor"></span>';
    const content = host.querySelector(".content");
    const segments = MOCK.insightText;
    let segIdx = 0, charIdx = 0;
    const charsPerTick = 2;
    const tick = 26; // ms

    function step() {
      if (segIdx >= segments.length) {
        host.querySelector(".typed-cursor").remove();
        return;
      }
      const seg = segments[segIdx];
      const target = seg.t;
      let span = content.lastElementChild;
      // Ensure we have a span for this segment
      if (!span || !span.dataset.segIdx || +span.dataset.segIdx !== segIdx) {
        span = document.createElement("span");
        span.dataset.segIdx = String(segIdx);
        if (seg.strong) span.innerHTML = "<strong></strong>";
        if (seg.risk) span.innerHTML = '<span class="risk"></span>';
        content.appendChild(span);
      }
      const inner = span.firstElementChild || span;
      const next = Math.min(target.length, charIdx + charsPerTick);
      inner.textContent = target.slice(0, next);
      charIdx = next;
      if (charIdx >= target.length) {
        segIdx++; charIdx = 0;
      }
      typingHandle = setTimeout(step, tick);
    }
    step();
  }

  /* Pulse the forecast crossing marker for emphasis at end. */
  function emphasizeCrossing() {
    const marker = document.querySelector(".forecast-chart .marker");
    if (!marker) return;
    marker.style.animation = "stationPulse 1.2s ease-out 3";
    marker.style.transformOrigin = "center";
  }

  /* Insight pipeline trail: data → forecast → reasoning → alert. */
  function trailProgress() {
    const trail = document.querySelector(".insight-trail");
    if (!trail) return;
    const steps = [...trail.querySelectorAll(".step")];
    const sequence = [
      [9.0,  0],  // data
      [10.0, 1],  // forecast
      [12.5, 2],  // reasoning
      [14.5, 3],  // alert
    ];
    sequence.forEach(([t, idx]) => {
      T(t, () => {
        steps.forEach((s, i) => {
          s.classList.remove("active");
          if (i < idx) s.classList.add("done");
          else if (i === idx) s.classList.add("active");
          else s.classList.add("pending");
        });
      });
    });
    T(15.5, () => {
      steps.forEach((s) => {
        s.classList.remove("active", "pending");
        s.classList.add("done");
      });
    });
  }

  /* Tick the topbar clock. */
  function clockTick() {
    const node = document.querySelector("[data-clock]");
    if (!node) return;
    function update() {
      const now = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      node.textContent = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
    }
    update();
    setInterval(update, 1000);
  }

  /* Animate the training meta numbers. */
  function tickTrainingMeta() {
    const epoch = document.querySelector("[data-meta=epoch]");
    const loss = document.querySelector("[data-meta=loss]");
    const lr = document.querySelector("[data-meta=lr]");
    const gpu = document.querySelector("[data-meta=gpu]");
    if (!epoch) return;

    let e = 0, l = 0.78;
    const step = () => {
      e = Math.min(80, e + 1);
      l = Math.max(0.061, l * 0.95 + 0.005 * (Math.random() - 0.5));
      epoch.textContent = `${e}/80`;
      loss.textContent = l.toFixed(4);
      lr.textContent = (3e-4 * (1 - e / 80)).toExponential(2).replace("e", "e");
      gpu.textContent = `${(74 + Math.random() * 8).toFixed(1)}%`;
    };
    const id = setInterval(step, 100);
    setTimeout(() => clearInterval(id), 9000);
  }

  function init() {
    /* Mount static structure */
    buildHero();
    Charts.drawSwissMap(document.querySelector(".map-frame"));
    Charts.drawLossCurve(document.querySelector(".loss-chart"));
    Charts.drawForecast(document.querySelector(".forecast-chart"));
    kpiPopulate();
    insightCardsPopulate();
    clockTick();

    /* Stage transitions */
    T(2.6, () => document.querySelector(".hero").classList.add("is-leaving"));
    T(3.4, () => {
      const hero = document.querySelector(".hero");
      if (hero) hero.style.display = "none";
    });

    /* Card reveal cascade */
    reveal(".card.data-agent", 3.0);
    reveal(".card.data-stream", 3.25);
    reveal(".card.data-kpi", 3.5);
    reveal(".card.training", 5.0);
    reveal(".card.hpo", 5.4);
    reveal(".card.insight", 8.4);
    reveal(".card.forecast", 8.7);

    /* Per-card content animations */
    streamPopulate();
    hpoPopulate();
    tickTrainingMeta();
    T(9.0, typeInsight);
    trailProgress();
    T(17.0, emphasizeCrossing);

    /* Production-ready badge */
    T(15.7, () => {
      const badge = document.querySelector(".ready-badge");
      if (badge) badge.classList.add("is-visible");
    });

    /* Optional loop for video recording */
    if (new URLSearchParams(location.search).get("loop") === "1") {
      T(20.0, () => location.reload());
    }
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", Scene.init);
