// ============================================================================
// Toast Notifications
// ============================================================================
function showToast(message, type = "info", duration = 3000) {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span>${message}</span>
    <button onclick="this.parentElement.remove()" class="toast-close">×</button>
  `;
  container.appendChild(toast);
  
  if (duration > 0) {
    setTimeout(() => toast.remove(), duration);
  }
}

// ============================================================================
// Utility Functions
// ============================================================================
function card(k, v) {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function setCards(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = rows.map((r) => card(r.k, r.v)).join("");
}

function table(id, rows) {
  const el = document.getElementById(id);
  if (!el) return;
  
  if (!rows || rows.length === 0) {
    el.innerHTML = "";
    return;
  }
  
  const cols = Object.keys(rows[0]);
  const thead = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const tbody = rows
    .map((r) => `<tr>${cols.map((c) => {
      const val = r[c];
      if (typeof val === "number") return `<td>${val.toFixed(6)}</td>`;
      if (typeof val === "boolean") return `<td>${val ? "YES" : "NO"}</td>`;
      return `<td>${val}</td>`;
    }).join("")}</tr>`)
    .join("");
  el.innerHTML = `<thead>${thead}</thead><tbody>${tbody}</tbody>`;
}

function parseFeatures() {
  const raw = document.getElementById("featureInput").value.trim();
  if (!raw) throw new Error("Feature input is empty.");
  
  const values = raw.split(/[\s,]+/).filter(Boolean).map(Number);
  if (values.some((v) => Number.isNaN(v))) {
    throw new Error("Feature input contains non-numeric values.");
  }
  
  return values;
}

async function post(url, body) {
  try {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Request failed");
    return j;
  } catch (e) {
    showToast(`Error: ${e.message}`, "error");
    throw e;
  }
}

function withLoading(btnId, fn) {
  const btn = document.getElementById(btnId);
  if (!btn) return fn();
  
  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "...";
  
  return fn()
    .finally(() => {
      btn.disabled = false;
      btn.textContent = originalText;
    });
}

// ============================================================================
// API Calls
// ============================================================================
async function loadStatus() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    const s = await fetch("/schema").then((r) => r.json());
    
    const epochEl = document.getElementById("epochStamp");
    if (epochEl) {
      epochEl.textContent = h.time || new Date().toISOString();
    }

    setCards("statusCards", [
      { k: "Service", v: h.status.toUpperCase() },
      { k: "Features", v: h.n_features },
      { k: "Alert Threshold", v: h.alert_threshold.toFixed(2) },
      { k: "Live Samples", v: h.live_samples },
      { k: "UAP Loaded", v: s.uap_loaded ? "✓ YES" : "✗ NO" },
      { k: "Dataset", v: s.simulation_dataset_loaded ? "✓ YES" : "✗ NO" },
    ]);
    
    showToast("Status updated", "ok", 1500);
  } catch (e) {
    showToast(`Failed to load status: ${e.message}`, "error");
  }
}

async function loadSample() {
  try {
    const idxRaw = document.getElementById("sampleIndex").value.trim();
    const url = idxRaw ? `/simulation_sample?index=${encodeURIComponent(idxRaw)}` : "/simulation_sample";
    
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Failed to load sample");

    document.getElementById("featureInput").value = j.features.join(", ");
    document.getElementById("inputInfo").textContent = 
      `✓ Loaded sample #${j.index} (Ground Truth: ${j.ground_truth_label}, Confidence Baseline: ready)`;
    
    showToast(`Sample #${j.index} loaded (${j.ground_truth_label})`, "ok");
  } catch (e) {
    showToast(`Failed to load sample: ${e.message}`, "error");
  }
}

async function runDefense() {
  try {
    const features = parseFeatures();
    const j = await post("/predict", { features });

    const alertBadge = j.alert 
      ? '<span class="badge warn">🚨 ALERT</span>' 
      : '<span class="badge ok">✓ SAFE</span>';
    
    setCards("defenseCards", [
      { k: "Prediction", v: j.prediction_label },
      { k: "Confidence", v: `${(j.confidence * 100).toFixed(2)}%` },
      { k: "Status", v: alertBadge },
      { k: "Timestamp", v: new Date(j.timestamp).toLocaleTimeString() },
    ]);
    
    if (j.alert) {
      showToast(`🚨 ALERT: ${j.prediction_label} (${(j.confidence * 100).toFixed(1)}%)`, "error", 5000);
    } else {
      showToast(`✓ Prediction: ${j.prediction_label}`, "ok");
    }
  } catch (e) {
    showToast(`Defense failed: ${e.message}`, "error");
  }
}

async function runAttack() {
  try {
    const features = parseFeatures();
    const epsilon = Number(document.getElementById("epsilon").value || "0.02");
    
    if (epsilon <= 0 || epsilon > 0.1) {
      throw new Error("Epsilon must be between 0.001 and 0.1");
    }
    
    const j = await post("/simulate_attack", { features, epsilon });

    const changedBadge = j.decision_changed
      ? '<span class="badge warn">⚠ CHANGED</span>'
      : '<span class="badge ok">✓ UNCHANGED</span>';

    setCards("attackCards", [
      { k: "Before", v: `${j.before.prediction_label} (${(j.before.confidence * 100).toFixed(1)}%)` },
      { k: "After", v: `${j.after.prediction_label} (${(j.after.confidence * 100).toFixed(1)}%)` },
      { k: "Decision", v: changedBadge },
      { k: "Epsilon", v: j.epsilon.toFixed(4) },
    ]);

    table("changesTable", j.top_feature_changes);
    
    if (j.decision_changed) {
      showToast(`⚠ Attack succeeded! Decision changed at ε=${epsilon.toFixed(4)}`, "error", 4000);
    } else {
      showToast(`✓ Defense held at ε=${epsilon.toFixed(4)}`, "ok");
    }
  } catch (e) {
    showToast(`Attack simulation failed: ${e.message}`, "error");
  }
}

async function loadAlerts() {
  try {
    const j = await fetch("/alerts?limit=50").then((r) => r.json());
    
    const countEl = document.getElementById("alertCount");
    if (countEl) {
      countEl.textContent = `${j.count} alerts`;
    }
    
    table("alertsTable", j.alerts || []);
    showToast(`Alerts refreshed (${j.count} total)`, "ok", 1500);
  } catch (e) {
    showToast(`Failed to load alerts: ${e.message}`, "error");
  }
}

async function checkDrift() {
  try {
    const j = await fetch("/drift?top_k=15").then((r) => r.json());
    
    if (j.status === "unavailable") {
      setCards("driftSummary", [
        { k: "Status", v: "Unavailable - No baseline" },
      ]);
      showToast("Drift monitoring unavailable (no reference baseline)", "error");
      return;
    }
    
    if (j.status === "insufficient_live_data") {
      setCards("driftSummary", [
        { k: "Live Samples", v: j.live_samples },
        { k: "Status", v: "Insufficient live data yet" },
      ]);
      showToast("Still collecting live samples for drift analysis", "info");
      return;
    }

    setCards("driftSummary", [
      { k: "Reference Samples", v: j.reference_samples },
      { k: "Live Samples", v: j.live_samples },
      { k: "Top Drifted", v: j.top_k },
      { k: "Status", v: j.status.toUpperCase() },
    ]);

    table("driftTable", j.top_drift_features || []);
    showToast(`Drift analysis complete (${j.top_k} features drifted)`, "ok");
  } catch (e) {
    showToast(`Drift check failed: ${e.message}`, "error");
  }
}

async function processBatch() {
  try {
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    
    if (!file) {
      throw new Error("Please select a CSV file first");
    }
    
    if (!file.name.toLowerCase().endsWith(".csv")) {
      throw new Error("File must be a CSV");
    }

    const formData = new FormData();
    formData.append("file", file);

    const r = await fetch("/predict_csv", {
      method: "POST",
      body: formData,
    });

    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "CSV processing failed");

    setCards("batchCards", [
      { k: "File", v: j.source },
      { k: "Total Samples", v: j.count },
      { k: "Attacks Detected", v: j.attack_count },
      { k: "Attack Rate", v: `${(j.attack_ratio * 100).toFixed(1)}%` },
    ]);

    table("batchResultsTable", j.records || []);
    showToast(`Batch processing complete: ${j.count} samples, ${j.attack_count} attacks`, "ok");
  } catch (e) {
    showToast(`Batch processing failed: ${e.message}`, "error");
  }
}

function clearBatch() {
  document.getElementById("csvFile").value = "";
  document.getElementById("batchCards").innerHTML = "";
  document.getElementById("batchResultsTable").innerHTML = "";
  document.getElementById("csvStatus").textContent = "No file selected";
  document.getElementById("csvLabel").classList.remove("has-file");
  showToast("Batch results cleared", "ok", 1500);
}

function handleCsvSelection() {
  const input = document.getElementById("csvFile");
  const status = document.getElementById("csvStatus");
  const label = document.getElementById("csvLabel");
  const file = input.files && input.files[0];

  if (!file) {
    status.textContent = "No file selected";
    label.classList.remove("has-file");
    return;
  }

  status.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  label.classList.add("has-file");
}

// ============================================================================
// Event Wiring
// ============================================================================
function wireEvents() {
  // Status & Monitoring
  document.getElementById("btnRefreshStatus").addEventListener("click", () => {
    withLoading("btnRefreshStatus", loadStatus);
  });

  document.getElementById("btnLoadSample").addEventListener("click", () => {
    withLoading("btnLoadSample", loadSample);
  });

  document.getElementById("btnLoadAlerts").addEventListener("click", () => {
    withLoading("btnLoadAlerts", loadAlerts);
  });

  document.getElementById("btnCheckDrift").addEventListener("click", () => {
    withLoading("btnCheckDrift", checkDrift);
  });

  // Defense & Attack
  document.getElementById("btnDefense").addEventListener("click", () => {
    withLoading("btnDefense", runDefense);
  });

  document.getElementById("btnAttack").addEventListener("click", () => {
    withLoading("btnAttack", runAttack);
  });

  // Batch & Drift
  document.getElementById("btnProcessCSV").addEventListener("click", () => {
    withLoading("btnProcessCSV", processBatch);
  });

  document.getElementById("btnClearBatch").addEventListener("click", clearBatch);

  document.getElementById("csvFile").addEventListener("change", handleCsvSelection);

  document.getElementById("btnRefreshDrift").addEventListener("click", () => {
    withLoading("btnRefreshDrift", checkDrift);
  });
}

// ============================================================================
// Initialization
// ============================================================================
window.addEventListener("DOMContentLoaded", async () => {
  wireEvents();
  await loadStatus().catch(() => {
    showToast("Failed to initialize - check if server is running", "error");
  });
  await loadAlerts().catch(() => {});
  
  // Auto-refresh status every 30 seconds
  setInterval(loadStatus, 30000);
});

