function card(k, v) {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function setCards(id, rows) {
  document.getElementById(id).innerHTML = rows.map((r) => card(r.k, r.v)).join("");
}

function table(id, rows) {
  const el = document.getElementById(id);
  if (!rows || rows.length === 0) {
    el.innerHTML = "";
    return;
  }
  const cols = Object.keys(rows[0]);
  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const tbody = rows
    .map((r) => `<tr>${cols.map((c) => `<td>${typeof r[c] === "number" ? r[c].toFixed(6) : r[c]}</td>`).join("")}</tr>`)
    .join("");
  el.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
}

function parseFeatures() {
  const raw = document.getElementById("featureInput").value.trim();
  if (!raw) throw new Error("Feature input is empty.");
  const values = raw.split(/[\s,]+/).filter(Boolean).map(Number);
  if (values.some((v) => Number.isNaN(v))) throw new Error("Feature input contains non-numeric values.");
  return values;
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}

async function loadStatus() {
  const h = await fetch("/health").then((r) => r.json());
  const s = await fetch("/schema").then((r) => r.json());
  const epochEl = document.getElementById("epochStamp");
  if (epochEl) {
    epochEl.textContent = h.time;
  }

  setCards("statusCards", [
    { k: "Service", v: h.status.toUpperCase() },
    { k: "Features", v: h.n_features },
    { k: "Alert Threshold", v: h.alert_threshold },
    { k: "UAP Loaded", v: s.uap_loaded ? "YES" : "NO" },
    { k: "Simulation Dataset", v: s.simulation_dataset_loaded ? "YES" : "NO" },
  ]);
}

async function loadSample() {
  const idxRaw = document.getElementById("sampleIndex").value.trim();
  const url = idxRaw ? `/simulation_sample?index=${encodeURIComponent(idxRaw)}` : "/simulation_sample";
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Failed to load sample");

  document.getElementById("featureInput").value = j.features.join(", ");
  document.getElementById("inputInfo").textContent = `Loaded sample #${j.index} (ground truth: ${j.ground_truth_label})`;
}

async function runDefense() {
  const features = parseFeatures();
  const j = await post("/predict", { features });

  const alertBadge = j.alert ? '<span class="badge warn">ALERT</span>' : '<span class="badge ok">NO ALERT</span>';
  setCards("defenseCards", [
    { k: "Prediction", v: j.prediction_label },
    { k: "Confidence", v: `${(j.confidence * 100).toFixed(2)}%` },
    { k: "Alert", v: alertBadge },
  ]);
}

async function runAttack() {
  const features = parseFeatures();
  const epsilon = Number(document.getElementById("epsilon").value || "0.02");
  const j = await post("/simulate_attack", { features, epsilon });

  const changedBadge = j.decision_changed
    ? '<span class="badge warn">CHANGED</span>'
    : '<span class="badge ok">UNCHANGED</span>';

  setCards("attackCards", [
    { k: "Before", v: `${j.before.prediction_label} (${(j.before.confidence * 100).toFixed(1)}%)` },
    { k: "After", v: `${j.after.prediction_label} (${(j.after.confidence * 100).toFixed(1)}%)` },
    { k: "Decision", v: changedBadge },
    { k: "Epsilon", v: j.epsilon },
  ]);

  table("changesTable", j.top_feature_changes);
}

async function loadAlerts() {
  const j = await fetch("/alerts?limit=50").then((r) => r.json());
  table("alertsTable", j.alerts || []);
}

function wireEvents() {
  document.getElementById("btnRefreshStatus").addEventListener("click", () => loadStatus().catch((e) => alert(e.message)));
  document.getElementById("btnLoadSample").addEventListener("click", () => loadSample().catch((e) => alert(e.message)));
  document.getElementById("btnDefense").addEventListener("click", () => runDefense().catch((e) => alert(e.message)));
  document.getElementById("btnAttack").addEventListener("click", () => runAttack().catch((e) => alert(e.message)));
  document.getElementById("btnLoadAlerts").addEventListener("click", () => loadAlerts().catch((e) => alert(e.message)));
}

window.addEventListener("DOMContentLoaded", async () => {
  wireEvents();
  await loadStatus();
  await loadAlerts();
});
