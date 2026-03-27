let FEATURE_NAMES = [];
let LAST_ATTACK_RESULT = null;

function card(title, value) {
  return `<div class="card"><div class="k">${title}</div><div class="v">${value}</div></div>`;
}

function setCards(elId, entries) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = entries.map((e) => card(e.k, e.v)).join("");
}

function renderTable(elId, rows, maxCols = null) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!rows || rows.length === 0) {
    el.innerHTML = "";
    return;
  }

  let cols = Object.keys(rows[0]);
  if (maxCols && cols.length > maxCols) {
    cols = cols.slice(0, maxCols);
  }

  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const tbody = rows
    .map((r) => {
      const cls = r.changed ? "changed" : "";
      return `<tr class="${cls}">${cols
        .map((c) => {
          const v = r[c];
          return `<td>${typeof v === "number" ? v.toFixed(6) : v}</td>`;
        })
        .join("")}</tr>`;
    })
    .join("");

  el.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}

function plotCurves(curves) {
  const x = curves.map((r) => r.epsilon);

  Plotly.newPlot(
    "chartAcc",
    [
      {
        x,
        y: curves.map((r) => r.accuracy),
        mode: "lines+markers",
        name: "Accuracy",
        line: { color: "#9be5ff", width: 3 },
      },
      {
        x,
        y: curves.map((r) => r.fnr),
        mode: "lines+markers",
        name: "FNR",
        line: { color: "#d59fff", width: 3 },
      },
      {
        x,
        y: curves.map((r) => r.fooling_rate),
        mode: "lines+markers",
        name: "PCC",
        line: { color: "#ffb2c6", width: 3 },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cfe6ff" },
      margin: { t: 30, l: 40, r: 16, b: 36 },
      xaxis: { title: "Epsilon", gridcolor: "rgba(140,170,220,0.15)" },
      yaxis: { title: "Score", gridcolor: "rgba(140,170,220,0.15)" },
      legend: { orientation: "h", y: 1.15 },
    },
    { displayModeBar: false }
  );
}

function plotMiniBars(elId, values, color) {
  Plotly.newPlot(
    elId,
    [
      {
        x: values.map((_, i) => i + 1),
        y: values,
        type: "bar",
        marker: { color },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { t: 10, l: 24, r: 10, b: 28 },
      font: { color: "#b7d4f6" },
      xaxis: { showgrid: false, zeroline: false },
      yaxis: { showgrid: true, gridcolor: "rgba(132,163,214,0.14)" },
    },
    { displayModeBar: false }
  );
}

function plotCM(elId, cm, title, colors) {
  Plotly.newPlot(
    elId,
    [{ z: cm, type: "heatmap", colorscale: colors, showscale: false }],
    {
      title,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cfe6ff" },
      xaxis: { title: "Predicted", tickvals: [0, 1], ticktext: ["Benign", "Malicious"] },
      yaxis: { title: "Actual", tickvals: [0, 1], ticktext: ["Benign", "Malicious"] },
      margin: { t: 42, l: 56, r: 20, b: 48 },
    },
    { displayModeBar: false }
  );
}

function plotFeatureImpact(rows) {
  Plotly.newPlot(
    "featureImpact",
    [
      {
        x: rows.map((r) => r.feature),
        y: rows.map((r) => r.abs_delta),
        type: "bar",
        marker: { color: rows.map((r) => r.abs_delta), colorscale: "PuRd" },
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cfe6ff" },
      margin: { t: 20, l: 46, r: 12, b: 80 },
      xaxis: { tickangle: -25 },
      yaxis: { gridcolor: "rgba(132,163,214,0.16)" },
    },
    { displayModeBar: false }
  );
}

function activateView(viewId) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
  const target = document.getElementById(viewId);
  if (target) target.classList.add("active");
  const btn = document.querySelector(`.nav-item[data-view="${viewId}"]`);
  if (btn) btn.classList.add("active");

  const titleMap = {
    dashboardView: "Adversarial IDS Attack Simulator",
    attackView: "Dataset & Operations",
    resultsView: "Post-Attack Evaluation",
    logsView: "Command Deck Telemetry",
  };
  const titleEl = document.getElementById("viewTitle");
  if (titleEl) titleEl.textContent = titleMap[viewId] || "Sentinel Prism";
}

async function loadInit() {
  const data = await fetch("/api/init").then((r) => r.json());
  FEATURE_NAMES = data.feature_names || [];

  setCards("datasetCards", [
    { k: "Corpus", v: "IDS-2024_Global" },
    { k: "Flow Records", v: `${(data.dataset_size / 1000).toFixed(1)}K` },
    { k: "Feature Vectors", v: data.n_features },
    { k: "Attack / Benign", v: `${data.attack_count} / ${data.benign_count}` },
  ]);

  renderTable("datasetTable", data.sample_rows || [], 12);
  if (document.getElementById("sampleIndex")) {
    document.getElementById("sampleIndex").max = Math.max(0, (data.test_size || 1) - 1);
  }

  // Simulated operations histogram from available dataset stats.
  const base = Math.max(4, Math.floor((data.test_size || 1000) / 500));
  const bars = [base, base + 1, base + 2, base + 4, base + 3, base + 2, base + 1, base + 5, base + 6, base + 4, base + 2];
  plotMiniBars("chartFnr", bars, "rgba(206,159,255,0.7)");
}

async function loadSample() {
  const sampleIndex = parseInt(document.getElementById("sampleIndex").value || "0", 10);
  const p = await postJSON("/api/predict", { sample_index: sampleIndex });

  const row = {};
  FEATURE_NAMES.forEach((f, i) => {
    row[f] = p.sample[i];
  });
  renderTable("sampleTable", [row], 10);
}

function renderDefaultKPIs(metrics) {
  setCards("metricsCards", [
    { k: "Accuracy Before Attack", v: `${(metrics.accuracy_before * 100).toFixed(2)}%` },
    { k: "Accuracy After Attack", v: `${(metrics.accuracy_after * 100).toFixed(2)}%` },
    { k: "False Negative Rate", v: `${(metrics.fnr_after * 100).toFixed(2)}%` },
    { k: "Attack Success Rate", v: `${(metrics.fooling_rate * 100).toFixed(2)}%` },
  ]);
}

async function applyAttack() {
  const sampleIndex = parseInt(document.getElementById("sampleIndex").value || "0", 10);
  const epsilon = parseFloat(document.getElementById("epsilon").value || "0.02");
  const evalSize = parseInt(document.getElementById("evalSize").value || "2000", 10);

  const a = await postJSON("/api/apply_attack", {
    sample_index: sampleIndex,
    epsilon,
    eval_size: evalSize,
  });

  LAST_ATTACK_RESULT = a;
  const epsilonLabel = document.getElementById("epsilonLabel");
  if (epsilonLabel) epsilonLabel.textContent = Number(epsilon).toFixed(3);

  renderTable("featureTable", a.feature_rows || [], 8);

  const changedBadge = a.decision_changed
    ? '<span class="badge danger">CHANGED</span>'
    : '<span class="badge success">UNCHANGED</span>';

  setCards("comparisonCards", [
    { k: "Pre-Attack", v: `${a.prediction_before_label} (${(a.confidence_before * 100).toFixed(1)}%)` },
    { k: "Post-Attack", v: `${a.prediction_after_label} (${(a.confidence_after * 100).toFixed(1)}%)` },
    { k: "Decision Shift", v: changedBadge },
    { k: "FNR Delta", v: `${((a.metrics.fnr_after - a.metrics.fnr_before) * 100).toFixed(2)}%` },
  ]);

  renderDefaultKPIs(a.metrics);
  plotCurves(a.curves || []);
  plotCM("cmBefore", a.cm_before, "Baseline Matrix", "Blues");
  plotCM("cmAfter", a.cm_after, "Post-Adversarial Matrix", "RdPu");
  plotFeatureImpact(a.feature_impact || []);

  // Logs view leverages the fooling curve for compact telemetry chart.
  Plotly.newPlot(
    "chartFool",
    [
      {
        x: (a.curves || []).map((r) => r.epsilon),
        y: (a.curves || []).map((r) => r.fooling_rate),
        mode: "lines+markers",
        line: { color: "#f9a8c5", width: 3 },
        name: "Attack Telemetry",
      },
    ],
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#cfe6ff" },
      margin: { t: 20, l: 46, r: 14, b: 42 },
      xaxis: { title: "Epsilon", gridcolor: "rgba(132,163,214,0.16)" },
      yaxis: { title: "Fooling Rate", gridcolor: "rgba(132,163,214,0.16)" },
    },
    { displayModeBar: false }
  );
}

function wireViewNavigation() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => activateView(btn.dataset.view));
  });
}

function wireControls() {
  const epsilonInput = document.getElementById("epsilon");
  if (epsilonInput) {
    epsilonInput.addEventListener("input", () => {
      const el = document.getElementById("epsilonLabel");
      if (el) el.textContent = Number(epsilonInput.value).toFixed(3);
    });
  }

  const btnSample = document.getElementById("btnLoadSample");
  if (btnSample) btnSample.addEventListener("click", () => loadSample().catch((e) => alert(e.message)));

  const btnAttack = document.getElementById("btnAttack");
  if (btnAttack) btnAttack.addEventListener("click", () => applyAttack().catch((e) => alert(e.message)));
}

window.addEventListener("DOMContentLoaded", async () => {
  wireViewNavigation();
  wireControls();
  activateView("dashboardView");

  try {
    await loadInit();
    await loadSample();
    await applyAttack();
  } catch (e) {
    alert(e.message);
  }
});
