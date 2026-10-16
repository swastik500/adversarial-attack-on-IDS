let LAST_UAP = null;
let BEFORE_SAMPLE = null;
let LAST_BENCHMARK = null;
let LAST_APPLY_RESULT = null;
let LAST_ATTACK_TYPE = 'none';
const BENCH_HISTORY_KEY = 'threatshield_bench_history_v1';
const UI_STATS = { total:0, ok:0, err:0, last:'-' };

const CICIDS_FEATURE_NAMES = [
  "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
  "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
  "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
  "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
  "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
  "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
  "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
  "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
  "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
  "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
  "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
  "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
  "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
  "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length Offset",
  "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
  "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
  "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
  "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
  "Active Max", "Active Min", "Idle Mean", "Idle Std",
  "Idle Max", "Idle Min"
];

function card(k,v,tone=''){return `<div class="card ${tone}"><div class="k">${k}</div><div class="v">${v}</div></div>`}
function setCards(id,items){document.getElementById(id).innerHTML=items.map(x=>card(x.k,x.v)).join('')}
function setCardsTone(id,items){document.getElementById(id).innerHTML=items.map(x=>card(x.k,x.v,x.tone||'')).join('')}

function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

function updateActivityStrip(){
  const total = document.getElementById('apTotal');
  const ok = document.getElementById('apOk');
  const err = document.getElementById('apErr');
  const last = document.getElementById('apLast');
  if(total) total.textContent = String(UI_STATS.total);
  if(ok) ok.textContent = String(UI_STATS.ok);
  if(err) err.textContent = String(UI_STATS.err);
  if(last) last.textContent = UI_STATS.last;
}

function showToast(message, tone='ok', timeoutMs=2600){
  const stack = document.getElementById('toastStack');
  if(!stack) return;
  const node = document.createElement('div');
  node.className = `toast ${tone}`;
  node.textContent = message;
  stack.prepend(node);
  setTimeout(()=>{
    node.style.opacity = '0';
    node.style.transform = 'translateY(6px)';
    setTimeout(()=>node.remove(), 180);
  }, timeoutMs);
}

async function withLoading(btnId, fn){
  const btn = document.getElementById(btnId);
  if(btn) btn.classList.add('loadingBtn');
  try{
    return await fn();
  }finally{
    if(btn) btn.classList.remove('loadingBtn');
  }
}

function confidenceBand(conf){
  if(conf>=0.9) return {label:'HIGH',tone:'bad'};
  if(conf>=0.7) return {label:'MEDIUM',tone:'warn'};
  return {label:'LOW',tone:'info'};
}

function prettyLabel(lbl){
  const v = String(lbl || '').toLowerCase();
  if(v === 'attack') return 'Attack Traffic';
  if(v === 'benign') return 'Normal Traffic';
  return v || 'Unknown';
}

function severityBadge(conf){
  const lvl = conf>=0.9 ? 'high' : (conf>=0.7 ? 'medium' : 'low');
  return `<span class="severity ${lvl}">${lvl.toUpperCase()}</span>`;
}

function setBackendStatus(ok, txt){
  const el = document.getElementById('backendStatus');
  if(!el) return;
  el.className = `statusChip ${ok ? 'ok' : 'bad'}`;
  el.textContent = txt;
}

function resetDemoTimeline(){
  for(let i=1;i<=6;i++){
    const el = document.getElementById(`demoStep${i}`);
    if(!el) continue;
    el.classList.remove('active','done');
  }
  const fill = document.getElementById('demoProgressFill');
  if(fill) fill.style.width = '0%';
  const now = document.getElementById('demoNow');
  if(now) now.textContent = 'Idle';
}

function setDemoTimelineState(stepIndex, label){
  for(let i=1;i<=6;i++){
    const el = document.getElementById(`demoStep${i}`);
    if(!el) continue;
    el.classList.toggle('done', i < stepIndex);
    el.classList.toggle('active', i === stepIndex);
  }
  const fill = document.getElementById('demoProgressFill');
  if(fill){
    const pct = Math.max(0, Math.min(100, Math.round((stepIndex - 1) / 6 * 100)));
    fill.style.width = `${pct}%`;
  }
  const now = document.getElementById('demoNow');
  if(now) now.textContent = label;
}

function completeDemoTimeline(){
  for(let i=1;i<=6;i++){
    const el = document.getElementById(`demoStep${i}`);
    if(!el) continue;
    el.classList.remove('active');
    el.classList.add('done');
  }
  const fill = document.getElementById('demoProgressFill');
  if(fill) fill.style.width = '100%';
  const now = document.getElementById('demoNow');
  if(now) now.textContent = 'Completed';
}

function parseSample(){
  const raw=document.getElementById('sampleInput').value.trim();
  if(!raw) throw new Error('Sample input is empty');
  const arr=raw.split(/[\s,]+/).filter(Boolean).map(Number);
  if(arr.some(Number.isNaN)) throw new Error('Sample has non-numeric value');
  return arr;
}

async function api(url,method='GET',body=null){
  const opts={method,headers:{'Content-Type':'application/json'}};
  if(body!==null) opts.body=JSON.stringify(body);
  UI_STATS.total += 1;
  UI_STATS.last = `${method} ${url}`;
  updateActivityStrip();

  const r=await fetch(url,opts);
  const j=await r.json();
  if(!r.ok){
    UI_STATS.err += 1;
    updateActivityStrip();
    throw new Error(j.detail||j.error||'Request failed');
  }
  UI_STATS.ok += 1;
  updateActivityStrip();
  return j;
}

function renderChartFallback(elId, xVals, yVals, title){
  const el = document.getElementById(elId);
  if(!el) return;
  if(!xVals.length || !yVals.length){
    el.innerHTML = `<div style="padding:12px;color:#b9cacb">No data for ${title}</div>`;
    return;
  }

  const w = Math.max(260, el.clientWidth || 260);
  const h = 240;
  const pad = 24;
  const minX = Math.min(...xVals), maxX = Math.max(...xVals);
  const minY = Math.min(...yVals), maxY = Math.max(...yVals);

  const sx = (v)=> pad + ((v - minX) / Math.max(1e-9, (maxX - minX))) * (w - pad*2);
  const sy = (v)=> (h - pad) - ((v - minY) / Math.max(1e-9, (maxY - minY))) * (h - pad*2);

  const pts = xVals.map((x,i)=>`${sx(x)},${sy(yVals[i])}`).join(' ');
  el.innerHTML = `
    <svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${w}" height="${h}" fill="#1d2023" />
      <text x="${pad}" y="16" fill="#e1e2e7" font-size="12">${title} (fallback)</text>
      <polyline points="${pts}" fill="none" stroke="#00f0ff" stroke-width="2" />
    </svg>
  `;
}

function renderMetricChart(elId, xVals, yVals, title){
  if(!Array.isArray(xVals) || !Array.isArray(yVals) || xVals.length===0 || yVals.length===0){
    renderChartFallback(elId, [], [], title);
    return;
  }

  if(typeof window.Plotly === 'undefined'){
    renderChartFallback(elId, xVals, yVals, title);
    return;
  }

  try{
    Plotly.newPlot(
      elId,
      [
        {x:xVals,y:yVals,mode:'lines+markers',name:title,line:{color:'#13d1b7',width:3},marker:{size:7,color:'#58e6ca'}},
      ],
      {
        title,
        paper_bgcolor:'#142027',
        plot_bgcolor:'#142027',
        font:{color:'#e7f2f4'},
        xaxis:{gridcolor:'#2b3f49'},
        yaxis:{gridcolor:'#2b3f49',range:[0,1]},
        margin:{t:40,l:40,r:20,b:40}
      },
      {displayModeBar:false,responsive:true}
    );
  }catch(_){
    renderChartFallback(elId, xVals, yVals, title);
  }
}

function renderBenchCharts(rows, useDefense){
  const names = rows.map(r=>String(r.attack_type || '').toUpperCase());
  const fool = rows.map(r=>Number(r.fooling_rate || 0));
  const fnr = rows.map(r=>Number(r.FNR_after || 0));

  renderMetricChart('benchFoolChart', names, fool, 'Fooling Rate by Attack');
  renderMetricChart('benchFnrChart', names, fnr, 'FNR by Attack');

  if(useDefense){
    if(typeof window.Plotly !== 'undefined'){
      try{
        Plotly.newPlot('benchFoolChart',[
          {x:names,y:fool,type:'bar',name:'No Defense',marker:{color:'#ff9973'}},
          {x:names,y:rows.map(r=>Number(r.defense_fooling_rate || 0)),type:'bar',name:'Defense',marker:{color:'#23c9b2'}}
        ],{
          barmode:'group',
          title:'Fooling Rate by Attack (Defense Comparison)',
          paper_bgcolor:'#142027',
          plot_bgcolor:'#142027',
          font:{color:'#e7f2f4'},
          xaxis:{gridcolor:'#2b3f49'},
          yaxis:{gridcolor:'#2b3f49',range:[0,1]},
          margin:{t:40,l:40,r:20,b:40}
        },{displayModeBar:false,responsive:true});
      }catch(_){ }
    }
  }
}

function renderDriftTable(rows){
  const t = document.getElementById('driftTable');
  if(!t) return;
  if(!Array.isArray(rows) || rows.length===0){
    t.innerHTML='<thead><tr><th>Status</th></tr></thead><tbody><tr><td>No drift data yet</td></tr></tbody>';
    return;
  }

  const cols = ['feature','reference_mean','live_mean','mean_abs_z','psi','drift_score'];
  const th = `<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const tb = rows.map(r=>`<tr>${cols.map(c=>{
    const v = r[c];
    return `<td>${typeof v === 'number' ? v.toFixed(5) : v}</td>`;
  }).join('')}</tr>`).join('');
  t.innerHTML = `<thead>${th}</thead><tbody>${tb}</tbody>`;
}

function renderChanges(mod,orig){
  const t = document.getElementById('changesTable');
  if(!t) return;
  if(!mod || !orig){
    t.innerHTML = '<tbody><tr><td style="color:var(--muted); text-align:center; padding: 20px;">No traffic sample perturbed yet. Start the simulation loop or apply an attack.</td></tr></tbody>';
    return;
  }
  
  const showOnlyPerturbed = document.getElementById('chkOnlyPerturbed') ? document.getElementById('chkOnlyPerturbed').checked : true;
  const rows = [];
  
  for(let i = 0; i < Math.min(mod.length, orig.length); i++){
    const d = mod[i] - orig[i];
    const absD = Math.abs(d);
    const changed = absD > 1e-9;
    
    if (showOnlyPerturbed && !changed) continue;
    
    const featName = CICIDS_FEATURE_NAMES[i] || `Feature ${i}`;
    rows.push({
      index: i,
      feature: featName,
      before: orig[i],
      after: mod[i],
      delta: d,
      changed: changed
    });
  }
  
  if(rows.length === 0){
    t.innerHTML = '<tbody><tr><td style="color:var(--muted); text-align:center; padding: 20px;">No perturbed features found for the current filter settings.</td></tr></tbody>';
    return;
  }
  
  const th = `
    <tr style="border-bottom:2px solid var(--line); color:var(--primary-2); font-size:0.8rem;">
      <th style="padding:10px 8px; text-align:left;">Index</th>
      <th style="padding:10px 8px; text-align:left;">Feature Name</th>
      <th style="padding:10px 8px; text-align:right;">Original Value</th>
      <th style="padding:10px 8px; text-align:right;">Perturbed Value</th>
      <th style="padding:10px 8px; text-align:right;">Delta (Δ)</th>
      <th style="padding:10px 8px; text-align:center;">Direction</th>
    </tr>
  `;
  
  const tb = rows.map(r => {
    const deltaColor = r.delta > 0 ? 'var(--danger)' : (r.delta < 0 ? 'var(--good)' : 'var(--muted)');
    const deltaSymbol = r.delta > 0 ? `+${r.delta.toFixed(6)}` : r.delta.toFixed(6);
    const directionHtml = r.delta > 0 
      ? '<span style="color:var(--danger); font-weight:700;">📈 INCREASED</span>' 
      : (r.delta < 0 ? '<span style="color:var(--good); font-weight:700;">📉 DECREASED</span>' : '<span style="color:var(--muted);">-</span>');
      
    return `
      <tr style="border-bottom:1px solid var(--line); font-size:0.85rem; ${r.changed ? 'background:#1a2730;' : ''}">
        <td style="padding:8px; color:var(--muted); font-family:monospace;">#${r.index}</td>
        <td style="padding:8px; font-weight:700; color:var(--text);">${r.feature}</td>
        <td style="padding:8px; text-align:right; font-family:monospace;">${r.before.toFixed(6)}</td>
        <td style="padding:8px; text-align:right; font-family:monospace; color:${r.changed ? 'var(--warning)' : 'inherit'}; font-weight:${r.changed ? '700' : 'normal'};">${r.after.toFixed(6)}</td>
        <td style="padding:8px; text-align:right; font-family:monospace; color:${deltaColor}; font-weight:700;">${deltaSymbol}</td>
        <td style="padding:8px; text-align:center;">${directionHtml}</td>
      </tr>
    `;
  }).join('');
  
  t.innerHTML = `<thead>${th}</thead><tbody>${tb}</tbody>`;
}

function renderBenchTable(rows, useDefense){
  const t=document.getElementById('benchTable');
  if(!Array.isArray(rows) || rows.length===0){
    t.innerHTML='<thead><tr><th>status</th></tr></thead><tbody><tr><td>No benchmark data</td></tr></tbody>';
    return;
  }

  const cols=['attack_type','fooling_rate','attack_to_benign_rate','accuracy_after','FNR_after'];
  if(useDefense){
    cols.push('defense_fooling_rate','defense_attack_to_benign_rate','defense_accuracy_after','defense_FNR_after');
  }

  const th=`<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const tb=rows.map(r=>`<tr>${cols.map(c=>{
    const v=r[c];
    if(typeof v==='number') return `<td>${(v*100).toFixed(2)}%</td>`;
    return `<td>${v ?? '-'}</td>`;
  }).join('')}</tr>`).join('');
  t.innerHTML=`<thead>${th}</thead><tbody>${tb}</tbody>`;
}

function getBenchHistory(){
  try{
    const raw = localStorage.getItem(BENCH_HISTORY_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  }catch(_){
    return [];
  }
}

function setBenchHistory(arr){
  localStorage.setItem(BENCH_HISTORY_KEY, JSON.stringify(arr.slice(0, 20)));
}

function renderBenchHistory(){
  const t = document.getElementById('benchHistoryTable');
  if(!t) return;
  const rows = getBenchHistory();
  if(rows.length===0){
    t.innerHTML = '<thead><tr><th>status</th></tr></thead><tbody><tr><td>No saved runs</td></tr></tbody>';
    return;
  }

  const cols = ['time','epsilon','sample_size','attack_types','base_accuracy','base_FNR','use_defense'];
  const th = `<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const tb = rows.map(r=>`<tr>${cols.map(c=>{
    const v = r[c];
    if(typeof v === 'number' && (c.includes('accuracy') || c.includes('FNR'))) return `<td>${(v*100).toFixed(2)}%</td>`;
    if(Array.isArray(v)) return `<td>${v.join(', ')}</td>`;
    return `<td>${v}</td>`;
  }).join('')}</tr>`).join('');
  t.innerHTML = `<thead>${th}</thead><tbody>${tb}</tbody>`;
}

function saveCurrentBenchmark(){
  if(!LAST_BENCHMARK) throw new Error('No benchmark run available. Run benchmark first.');
  const hist = getBenchHistory();
  const row = {
    time: new Date().toLocaleString(),
    epsilon: LAST_BENCHMARK.epsilon,
    sample_size: LAST_BENCHMARK.sample_size,
    attack_types: (LAST_BENCHMARK.results || []).map(r=>r.attack_type),
    base_accuracy: LAST_BENCHMARK.base_accuracy,
    base_FNR: LAST_BENCHMARK.base_FNR,
    use_defense: LAST_BENCHMARK.use_defense ? 'yes' : 'no',
  };
  hist.unshift(row);
  setBenchHistory(hist);
  renderBenchHistory();
}

function clearBenchHistory(){
  localStorage.removeItem(BENCH_HISTORY_KEY);
  renderBenchHistory();
}

function exportBenchmarkCsv(){
  if(!LAST_BENCHMARK || !Array.isArray(LAST_BENCHMARK.results) || LAST_BENCHMARK.results.length===0){
    throw new Error('No benchmark results to export. Run benchmark first.');
  }
  const rows = LAST_BENCHMARK.results;
  const headers = Object.keys(rows[0]);
  const esc = (v)=>`"${String(v ?? '').replace(/"/g,'""')}"`;
  const lines = [headers.join(',')];
  for(const r of rows){
    lines.push(headers.map(h=>esc(r[h])).join(','));
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `attack_benchmark_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function refreshBackendHealth(){
  try{
    const h = await api('/health');
    setBackendStatus(true, `Backend: ${h.status} | Uptime ${h.uptime_sec}s | Device ${h.device}`);
  }catch(_){
    setBackendStatus(false, 'Backend: offline/unreachable');
  }
}

async function loadSample(kind){
  const j = await api(`/sample?kind=${encodeURIComponent(kind)}`);
  BEFORE_SAMPLE = j.sample;
}

async function loadRandomSample(){
  const kind = Math.random() > 0.5 ? 'attack' : 'benign';
  const j = await api(`/sample?kind=${kind}`);
  BEFORE_SAMPLE = j.sample;
  return j.sample;
}

async function doPredict(){
  const s = BEFORE_SAMPLE || await loadRandomSample();
  const j = await api('/predict','POST',{sample:s});
  LAST_APPLY_RESULT = j;
  LAST_ATTACK_TYPE = 'none';
  updateDefenseDashboard(j, 'none', document.getElementById('globalDefenseToggle').checked);
  showToast('Defense check completed', 'ok');
}

async function doGenerate(){
  const eps=Number(document.getElementById('epsilon').value);
  const j=await api('/generate-attack','POST',{epsilon:eps});
  LAST_UAP=j.uap;
  showToast('Perturbation generated', 'warn');
}

async function doApply(){
  const s = BEFORE_SAMPLE || await loadRandomSample();
  const attackType = document.getElementById('attackSelect').value;
  const eps = Number(document.getElementById('epsilon').value);
  const defenseActive = document.getElementById('globalDefenseToggle').checked;
  const body = {sample: s, attack_type: attackType, epsilon: eps, defense_enabled: defenseActive};
  const j = await api('/apply-attack','POST',body);
  LAST_APPLY_RESULT = j;
  LAST_ATTACK_TYPE = attackType;
  updateDefenseDashboard(j, attackType, defenseActive);
  recordAttackHistory(j, attackType, eps, defenseActive);
  showToast('Applied evasion perturbation', 'ok');
}

function updateDefenseDashboard(response, attackType, defenseActive) {
  let j = response;
  let attributions = null;
  if (response && response.feature_attributions) {
    attributions = response.feature_attributions;
  }
  if (j && j.new_prediction) {
    j = j.new_prediction;
  }
  const isWhite = ['fgsm', 'bim', 'pgd', 'uap'].includes(attackType);
  const isBlack = ['query'].includes(attackType);
  const isNoise = ['random'].includes(attackType);
  
  let typeStr = '-';
  if(attackType !== 'none') {
    typeStr = `${attackType} (${isWhite ? 'White-Box' : (isBlack ? 'Black-Box' : 'Noise')})`;
  } else {
    typeStr = 'none (Clean)';
  }
  
  document.getElementById('defCardAttackType').textContent = typeStr.toUpperCase();
  
  const confidence = j.confidence || 0.0;
  document.getElementById('defCardConf').textContent = `${(confidence * 100).toFixed(2)}%`;
  
  // Dynamic Trained Model Used override mapping
  const activeModelSelected = document.getElementById('defActiveModelSelector').value;
  const modelKeyMap = {
    dqn_clean: "DQN (Clean)",
    dqn_adv: "DQN (Adv-Trained)",
    mlp: "MLP",
    cnn: "CNN",
    rnn: "RNN",
    gbt: "GBT"
  };
  const isAuto = activeModelSelected === 'auto';
  const modelKey = modelKeyMap[activeModelSelected];
  const modelPred = j.models && j.models[modelKey];

  let statusStr = '-';
  let statusColor = 'var(--good)';
  if (attackType === 'none') {
    statusStr = 'SECURE (Normal Profile)';
  } else {
    if (defenseActive) {
      if (isAuto) {
        if (j.anomaly && j.anomaly.is_anomaly) {
          statusStr = '🚨 BLOCKED (VAE Anomaly)';
          statusColor = 'var(--primary-2)';
        } else {
          const advDqn = j.models && j.models["DQN (Adv-Trained)"];
          if (advDqn && advDqn.label === 'attack') {
            statusStr = '🛡️ NEUTRALIZED (DQN Robust)';
            statusColor = 'var(--primary)';
          } else {
            statusStr = '🚨 COMPROMISED (Bypassed)';
            statusColor = 'var(--danger)';
          }
        }
      } else {
        // Individual model defense evaluation
        if (modelPred && modelPred.label === 'attack') {
          statusStr = `🛡️ NEUTRALIZED (${modelKey})`;
          statusColor = 'var(--primary)';
        } else {
          statusStr = `🚨 COMPROMISED (${modelKey})`;
          statusColor = 'var(--danger)';
        }
      }
    } else {
      statusStr = '🚨 COMPROMISED (Shield Off)';
      statusColor = 'var(--danger)';
    }
  }
  const statusEl = document.getElementById('defCardStatus');
  statusEl.textContent = statusStr;
  statusEl.style.color = statusColor;

  document.getElementById('defCardTimestamp').textContent = new Date().toLocaleTimeString();
  document.getElementById('defCardMode').textContent = attackType === 'none' ? 'N/A' : (isWhite ? 'WHITE-BOX' : (isBlack ? 'BLACK-BOX' : 'NOISE'));
  
  // Set the text value of active model used
  const labelEl = document.getElementById('defCardModel');
  if(labelEl) {
    labelEl.textContent = isAuto ? (j.anomaly && j.anomaly.is_anomaly ? 'VAE Detector' : 'DQN (Adv-Trained)') : modelKey;
  }

  const bodyEl = document.getElementById('crossModelBody');
  if (bodyEl && j.models) {
    bodyEl.innerHTML = '';
    const trueLabel = attackType === 'none' ? 'benign' : 'attack';
    let bestModelName = null;
    let highestF1 = -1.0;
    
    for (const [modelName, mData] of Object.entries(j.models)) {
      if (mData.label === trueLabel && mData.f1 > highestF1) {
        highestF1 = mData.f1;
        bestModelName = modelName;
      }
    }
    if (!bestModelName) {
      for (const [modelName, mData] of Object.entries(j.models)) {
        if (mData.f1 > highestF1) {
          highestF1 = mData.f1;
          bestModelName = modelName;
        }
      }
    }

    for (const [modelName, mData] of Object.entries(j.models)) {
      const isBest = modelName === bestModelName;
      const predText = mData.label.toUpperCase();
      const predColor = mData.label === 'attack' ? 'var(--danger)' : 'var(--good)';
      const confText = `${(mData.confidence * 100).toFixed(1)}%`;
      const rowClass = isBest ? 'class="bestModelRow"' : '';
      
      bodyEl.innerHTML += `
        <tr ${rowClass} style="border-bottom:1px solid var(--line); font-size:0.85rem;">
          <td style="padding:10px 6px; font-weight:700;">${modelName} ${isBest ? '🏆' : ''}</td>
          <td style="padding:10px 6px; color:${predColor}; font-weight:700;">${predText}</td>
          <td style="padding:10px 6px; font-family:monospace;">${confText}</td>
          <td style="padding:10px 6px; color:var(--muted);">${(mData.accuracy * 100).toFixed(1)}%</td>
          <td style="padding:10px 6px; color:var(--muted);">${(mData.precision * 100).toFixed(1)}%</td>
          <td style="padding:10px 6px; color:var(--muted);">${(mData.recall * 100).toFixed(1)}%</td>
          <td style="padding:10px 6px; color:var(--primary-2); font-weight:700;">${mData.f1.toFixed(3)}</td>
          <td style="padding:10px 6px; font-family:monospace; color:var(--warning);">${mData.inference_time_ms.toFixed(2)} ms</td>
        </tr>
      `;
    }
  }

  const alertEl = document.getElementById('visualStatusAlert');
  if(alertEl) {
    if(attackType === 'none') {
      alertEl.className = 'visualAlert normal';
      alertEl.innerHTML = `<span>ℹ️ SYSTEM ACTIVE: Monitoring clean network traffic. No active perturbation.</span>`;
    } else {
      if(!defenseActive) {
        alertEl.className = 'visualAlert compromised';
        alertEl.innerHTML = `<span>🚨 SYSTEM COMPROMISED: ${attackType.toUpperCase()} intrusion bypassed the deactivated classifier!</span>`;
      } else {
        if (isAuto) {
          if(j.anomaly && j.anomaly.is_anomaly) {
            alertEl.className = 'visualAlert protected';
            alertEl.innerHTML = `<span>🛡️ INTRUSION BLOCKED: VAE Anomaly Check flagged the perturbation (MSE: ${j.anomaly.reconstruction_error.toFixed(6)} > 0.05)!</span>`;
          } else {
            const advDqn = j.models && j.models["DQN (Adv-Trained)"];
            if(advDqn && advDqn.label === 'attack') {
              alertEl.className = 'visualAlert neutralized';
              alertEl.innerHTML = `<span>🛡️ ATTACK NEUTRALIZED: Adversarially trained DQN policy correctly identified the attack profile!</span>`;
            } else {
              alertEl.className = 'visualAlert compromised';
              alertEl.innerHTML = `<span>🚨 ATTACK BYPASSED: Intrusion successfully fooled the active defense layers.</span>`;
            }
          }
        } else {
          // Alert message for custom classifier defense
          if(modelPred && modelPred.label === 'attack') {
            alertEl.className = 'visualAlert neutralized';
            alertEl.innerHTML = `<span>🛡️ ATTACK NEUTRALIZED: Active protection classifier (${modelKey}) correctly identified the attack profile!</span>`;
          } else {
            alertEl.className = 'visualAlert compromised';
            alertEl.innerHTML = `<span>🚨 ATTACK BYPASSED: Intrusion successfully fooled the active classifier (${modelKey})!</span>`;
          }
        }
      }
    }
  }

  const xaiSection = document.getElementById('xaiSection');
  const xaiAttrsEl = document.getElementById('xaiAttributions');
  if (xaiSection && xaiAttrsEl) {
    if (attributions && attributions.length > 0) {
      xaiSection.style.display = 'block';
      
      const mapped = attributions.map((v, i) => ({
        index: i,
        name: CICIDS_FEATURE_NAMES[i] || `Feature ${i}`,
        value: v
      }));
      mapped.sort((a, b) => b.value - a.value);
      
      const top5 = mapped.slice(0, 5);
      
      xaiAttrsEl.innerHTML = top5.map(item => {
        const pct = (item.value * 100).toFixed(1);
        return `
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; background: var(--bg-2); padding: 8px 12px; border-radius: 4px; border: 1px solid var(--line);">
            <div style="flex: 1; min-width: 0; padding-right: 15px; text-align: left;">
              <span style="color: var(--muted); font-family: monospace; font-size: 0.75rem; margin-right: 6px;">#${item.index}</span>
              <strong style="color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.name}</strong>
            </div>
            <div style="width: 140px; display: flex; align-items: center; gap: 8px;">
              <div style="flex: 1; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden;">
                <div style="width: ${pct}%; height: 100%; background: linear-gradient(90deg, var(--warning), var(--danger)); border-radius: 3px;"></div>
              </div>
              <span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: var(--warning);">${pct}%</span>
            </div>
          </div>
        `;
      }).join('');
    } else {
      xaiSection.style.display = 'none';
    }
  }
}

let simulationTimer = null;
let simulationPackets = 0;
let simulationStart = null;
let cachedTestSamples = [];

async function getCachedSample() {
  if (cachedTestSamples.length === 0) {
    for (let i = 0; i < 5; i++) {
      const res = await api('/sample?kind=attack');
      cachedTestSamples.push(res.sample);
    }
  }
  const idx = Math.floor(Math.random() * cachedTestSamples.length);
  return cachedTestSamples[idx];
}

async function startAttackSimulation() {
  if (simulationTimer) return;
  
  document.getElementById('btnStartSimulation').disabled = true;
  document.getElementById('btnStopSimulation').disabled = false;
  
  const statusEl = document.getElementById('simCardStatus');
  if(statusEl) statusEl.innerHTML = '<span class="statusBadge running">RUNNING</span>';
  
  simulationPackets = 0;
  simulationStart = Date.now();
  
  showToast('Real-time evasion attack simulation started', 'err');
  
  simulationTimer = setInterval(async () => {
    try {
      const s = await getCachedSample();
      const attackType = document.getElementById('attackSelect').value;
      const eps = Number(document.getElementById('epsilon').value);
      const defenseActive = document.getElementById('globalDefenseToggle').checked;
      
      const body = {sample: s, attack_type: attackType, epsilon: eps, defense_enabled: defenseActive};
      const j = await api('/apply-attack', 'POST', body);
      LAST_APPLY_RESULT = j;
      LAST_ATTACK_TYPE = attackType;
      
      simulationPackets++;
      document.getElementById('simCardPackets').textContent = simulationPackets;
      
      const dur = ((Date.now() - simulationStart) / 1000).toFixed(1) + 's';
      document.getElementById('simCardDuration').textContent = dur;
      
      updateDefenseDashboard(j, attackType, defenseActive);
      recordAttackHistory(j, attackType, eps, defenseActive);
      
    } catch(e) {
      console.warn("Simulation tick error:", e);
    }
  }, 900);
}

function stopAttackSimulation() {
  if (!simulationTimer) return;
  clearInterval(simulationTimer);
  simulationTimer = null;
  
  document.getElementById('btnStartSimulation').disabled = false;
  document.getElementById('btnStopSimulation').disabled = true;
  
  const statusEl = document.getElementById('simCardStatus');
  if(statusEl) statusEl.innerHTML = '<span class="statusBadge stopped">STOPPED</span>';
  
  showToast('Evasion simulation stopped', 'ok');
}

/* --- Attack Simulation History Tracking & Multi-Format Reporting --- */
let attackHistory = [];
const MAX_ATTACK_HISTORY = 100;

function evaluateAttackOutcome(response, attackType, defenseActive) {
  try {
    let j = response;
    if (j && j.new_prediction) {
      j = j.new_prediction;
    }
    const modelSelector = document.getElementById('defActiveModelSelector');
    const activeModelSelected = modelSelector ? modelSelector.value : 'auto';
    const isAuto = activeModelSelected === 'auto';
    
    const modelKeyMap = {
      dqn_clean: "DQN (Clean)",
      dqn_adv: "DQN (Adv-Trained)",
      mlp: "MLP",
      cnn: "CNN",
      rnn: "RNN",
      gbt: "GBT"
    };
    const modelKey = modelKeyMap[activeModelSelected] || "DQN (Clean)";
    const modelPred = j && j.models ? j.models[modelKey] : null;

    if (attackType === 'none') {
      return { outcomeText: 'CLEAN PASS', isBypassed: false, badgeClass: 'blocked', icon: 'ℹ️' };
    }

    if (!defenseActive) {
      return { outcomeText: 'BYPASS (Shield Off)', isBypassed: true, badgeClass: 'evaded', icon: '🚨' };
    }

    if (isAuto) {
      if (j && j.anomaly && j.anomaly.is_anomaly) {
        return { outcomeText: 'BLOCKED (VAE Anomaly)', isBypassed: false, badgeClass: 'blocked', icon: '🛡️' };
      }
      const advDqn = j && j.models && j.models["DQN (Adv-Trained)"];
      if (advDqn && advDqn.label === 'attack') {
        return { outcomeText: 'NEUTRALIZED (DQN)', isBypassed: false, badgeClass: 'neutralized', icon: '🛡️' };
      }
      return { outcomeText: 'BYPASS (Evaded)', isBypassed: true, badgeClass: 'evaded', icon: '🚨' };
    } else {
      if (modelPred && modelPred.label === 'attack') {
        return { outcomeText: `NEUTRALIZED (${modelKey})`, isBypassed: false, badgeClass: 'neutralized', icon: '🛡️' };
      }
      return { outcomeText: `BYPASS (${modelKey})`, isBypassed: true, badgeClass: 'evaded', icon: '🚨' };
    }
  } catch (err) {
    console.warn("evaluateAttackOutcome fallback due to:", err);
    return { outcomeText: 'SIMULATED ATTACK', isBypassed: true, badgeClass: 'evaded', icon: '⚡' };
  }
}

function recordAttackHistory(response, attackType, epsilon, defenseActive) {
  try {
    if (!response) return;
    const outcome = evaluateAttackOutcome(response, attackType, defenseActive);
    
    const origPred = response.original_prediction || {};
    const newPred = response.new_prediction || response;
    
    const cleanLabel = (origPred.label_name || (origPred.label_id === 1 ? 'attack' : (origPred.label_id === 0 ? 'benign' : 'attack'))).toLowerCase();
    const cleanConf = typeof origPred.confidence === 'number' ? origPred.confidence : (origPred.label_id === 1 ? 0.98 : 0.95);
    
    const advLabel = (newPred.label_name || (newPred.label_id === 1 ? 'attack' : (newPred.label_id === 0 ? 'benign' : 'unknown'))).toLowerCase();
    const advConf = typeof newPred.confidence === 'number' ? newPred.confidence : 0.5;

    const isWhite = ['fgsm', 'bim', 'pgd', 'uap'].includes(attackType);
    const isBlack = ['query'].includes(attackType);
    const vectorCategory = isWhite ? 'White-Box' : (isBlack ? 'Black-Box' : 'Noise');
    
    const record = {
      id: attackHistory.length + 1,
      time: new Date().toLocaleTimeString(),
      timestamp: Date.now(),
      vector: (attackType || 'UNKNOWN').toUpperCase(),
      vectorCategory: vectorCategory,
      epsilon: Number(epsilon || 0.02).toFixed(3),
      defenseActive: !!defenseActive,
      cleanLabel: cleanLabel,
      cleanConf: cleanConf,
      advLabel: advLabel,
      advConf: advConf,
      outcomeText: outcome.outcomeText,
      icon: outcome.icon,
      isBypassed: outcome.isBypassed,
      badgeClass: outcome.badgeClass,
      confDelta: advConf - cleanConf
    };

    attackHistory.unshift(record);
    if (attackHistory.length > MAX_ATTACK_HISTORY) {
      attackHistory.pop();
    }

    try {
      localStorage.setItem('threatshield_attack_history', JSON.stringify(attackHistory));
    } catch(e) {}

    renderAttackHistory();
  } catch (err) {
    console.error("Error recording attack history:", err);
  }
}

function renderAttackHistory() {
  const bodyEl = document.getElementById('attackHistoryBody');
  const countEl = document.getElementById('attackHistoryCount');
  const statTotalEl = document.getElementById('histStatTotal');
  const statBypassedEl = document.getElementById('histStatBypassed');
  const statBlockedEl = document.getElementById('histStatBlocked');
  const statRateEl = document.getElementById('histStatRate');

  if (!bodyEl) return;

  if (countEl) countEl.textContent = `${attackHistory.length} Recorded`;

  const total = attackHistory.length;
  const bypassedCount = attackHistory.filter(x => x.isBypassed).length;
  const blockedCount = total - bypassedCount;
  const evasionRate = total > 0 ? ((bypassedCount / total) * 100).toFixed(1) : '0.0';

  if (statTotalEl) statTotalEl.textContent = total;
  if (statBypassedEl) statBypassedEl.textContent = bypassedCount;
  if (statBlockedEl) statBlockedEl.textContent = blockedCount;
  if (statRateEl) statRateEl.textContent = `${evasionRate}%`;

  if (attackHistory.length === 0) {
    bodyEl.innerHTML = `
      <tr>
        <td colspan="7" style="text-align:center; padding:18px 8px; color:var(--muted); font-style:italic; font-size:0.8rem;">
          No attack simulations recorded yet. Launch an attack simulation above to start recording history.
        </td>
      </tr>
    `;
    return;
  }

  bodyEl.innerHTML = attackHistory.map((item) => {
    const deltaVal = item.confDelta;
    const deltaSign = deltaVal > 0 ? '+' : '';
    const deltaColor = item.isBypassed ? 'var(--danger)' : 'var(--good)';
    const cleanLabelColor = item.cleanLabel === 'attack' ? 'var(--danger)' : 'var(--good)';
    const advLabelColor = item.advLabel === 'attack' ? 'var(--danger)' : 'var(--good)';

    return `
      <tr class="histRow" style="border-bottom:1px solid var(--line); font-size:0.78rem;">
        <td style="padding:6px 6px; font-family:monospace; color:var(--muted);">#${item.id}</td>
        <td style="padding:6px 6px; font-family:monospace; color:var(--muted);">${item.time}</td>
        <td style="padding:6px 6px;">
          <strong style="color:var(--text);">${item.vector}</strong>
          <span style="font-size:0.66rem; color:var(--muted); display:block;">${item.vectorCategory}</span>
        </td>
        <td style="padding:6px 6px; font-family:monospace; color:var(--warning); font-weight:700;">${item.epsilon}</td>
        <td style="padding:6px 6px;">
          <span style="color:${cleanLabelColor}; font-weight:700;">${item.cleanLabel.toUpperCase()}</span>
          <span style="color:var(--muted); font-size:0.7rem;">(${(item.cleanConf * 100).toFixed(0)}%)</span>
          <span style="color:var(--muted); margin:0 2px;">➔</span>
          <span style="color:${advLabelColor}; font-weight:700;">${item.advLabel.toUpperCase()}</span>
          <span style="color:var(--muted); font-size:0.7rem;">(${(item.advConf * 100).toFixed(0)}%)</span>
        </td>
        <td style="padding:6px 6px;">
          <span class="histBadge ${item.badgeClass}">${item.icon} ${item.outcomeText}</span>
        </td>
        <td style="padding:6px 6px; font-family:monospace; font-weight:700; color:${deltaColor};">
          ${deltaSign}${(deltaVal * 100).toFixed(1)}%
        </td>
      </tr>
    `;
  }).join('');
}

function initAttackHistory() {
  try {
    const saved = localStorage.getItem('threatshield_attack_history');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed)) {
        attackHistory = parsed;
      }
    }
  } catch(e) {
    console.warn("Could not load stored attack history:", e);
  }
  renderAttackHistory();
}

function clearAttackHistory() {
  attackHistory = [];
  try {
    localStorage.removeItem('threatshield_attack_history');
  } catch(e) {}
  renderAttackHistory();
  showToast('Attack history cleared', 'ok');
}

function exportAttackHistoryJson() {
  if (attackHistory.length === 0) {
    showToast('No attack history to export', 'warn');
    return;
  }
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(attackHistory, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `threatshield_attack_history_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
  showToast(`Exported ${attackHistory.length} attack records as JSON`, 'ok');
}

function exportAttackHistoryCsv() {
  if (attackHistory.length === 0) {
    showToast('No attack history to export', 'warn');
    return;
  }
  const headers = [
    "ID",
    "Timestamp",
    "Vector",
    "Category",
    "Epsilon",
    "Original_Label",
    "Original_Confidence",
    "Attacked_Label",
    "Attacked_Confidence",
    "Confidence_Delta",
    "Outcome",
    "Is_Bypassed",
    "Defense_Active"
  ];
  const rows = attackHistory.map(item => [
    item.id,
    `"${item.time}"`,
    `"${item.vector}"`,
    `"${item.vectorCategory}"`,
    item.epsilon,
    `"${item.cleanLabel}"`,
    (item.cleanConf * 100).toFixed(2) + "%",
    `"${item.advLabel}"`,
    (item.advConf * 100).toFixed(2) + "%",
    (item.confDelta * 100).toFixed(2) + "%",
    `"${item.outcomeText.replace(/"/g, '""')}"`,
    item.isBypassed ? "YES" : "NO",
    item.defenseActive ? "ON" : "OFF"
  ]);

  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `threatshield_attack_history_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  showToast(`Exported ${attackHistory.length} records as CSV spreadsheet`, 'ok');
}

function exportAttackHistoryTxt() {
  if (attackHistory.length === 0) {
    showToast('No attack history to export', 'warn');
    return;
  }
  const total = attackHistory.length;
  const bypassed = attackHistory.filter(x => x.isBypassed).length;
  const blocked = total - bypassed;
  const rate = ((bypassed / total) * 100).toFixed(1);

  let txt = `=======================================================\n`;
  txt += `THREATSHIELD IDS - ADVERSARIAL ATTACK SIMULATION REPORT\n`;
  txt += `Generated At : ${new Date().toLocaleString()}\n`;
  txt += `Total Attacks: ${total} | Evasions: ${bypassed} | Blocked: ${blocked} | Evasion Rate: ${rate}%\n`;
  txt += `=======================================================\n\n`;
  txt += `DETAILED ACTIVITY LOGS:\n`;
  txt += `-------------------------------------------------------\n`;

  attackHistory.forEach(item => {
    txt += `[#${item.id}] ${item.time} | Vector: ${item.vector} (${item.vectorCategory}) | Epsilon: ${item.epsilon}\n`;
    txt += `     Original : ${item.cleanLabel.toUpperCase()} (${(item.cleanConf * 100).toFixed(1)}%)\n`;
    txt += `     Attacked : ${item.advLabel.toUpperCase()} (${(item.advConf * 100).toFixed(1)}%)\n`;
    txt += `     Outcome  : ${item.outcomeText} [Delta: ${(item.confDelta * 100).toFixed(1)}%] | Shield: ${item.defenseActive ? 'ON' : 'OFF'}\n`;
    txt += `-------------------------------------------------------\n`;
  });

  const blob = new Blob([txt], { type: "text/plain;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `threatshield_attack_history_${Date.now()}.txt`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  showToast(`Exported ${attackHistory.length} records as TXT summary report`, 'ok');
}

async function triggerSingleAttack() {
  try {
    const s = await getCachedSample();
    const attackType = document.getElementById('attackSelect').value;
    const eps = Number(document.getElementById('epsilon').value);
    const defenseActive = document.getElementById('globalDefenseToggle').checked;
    
    const body = {sample: s, attack_type: attackType, epsilon: eps, defense_enabled: defenseActive};
    const j = await api('/apply-attack', 'POST', body);
    LAST_APPLY_RESULT = j;
    LAST_ATTACK_TYPE = attackType;
    
    simulationPackets++;
    document.getElementById('simCardPackets').textContent = simulationPackets;
    
    updateDefenseDashboard(j, attackType, defenseActive);
    recordAttackHistory(j, attackType, eps, defenseActive);
    showToast(`Simulated 1 ${attackType.toUpperCase()} attack packet`, 'ok');
  } catch(e) {
    console.warn("Single attack error:", e);
    handleUiError(e);
  }
}

function renderCombinedMetricsChart(xVals, drVals, farVals, abrVals, fnrVals, foolingVals) {
  const elId = 'combinedMetricsChart';
  const el = document.getElementById(elId);
  if(!el) return;

  if (typeof window.Plotly === 'undefined') {
    el.innerHTML = '<div style="padding:12px;color:#b9cacb">Plotly is not loaded.</div>';
    return;
  }

  try {
    const traceDR = {
      x: xVals,
      y: drVals,
      mode: 'lines+markers',
      name: 'Intrusion Detection Rate (DR) [Higher is Better]',
      line: { color: '#23c9b2', width: 3 },
      marker: { size: 6 }
    };
    const traceFAR = {
      x: xVals,
      y: farVals,
      mode: 'lines+markers',
      name: 'False Alarm Rate (FAR) [Lower is Better]',
      line: { color: '#ffb85c', width: 3 },
      marker: { size: 6 }
    };
    const traceABR = {
      x: xVals,
      y: abrVals,
      mode: 'lines+markers',
      name: 'Adversarial Bypass Rate (ABR) [Lower is Better]',
      line: { color: '#ff7f7f', width: 3 },
      marker: { size: 6 }
    };
    const traceFNR = {
      x: xVals,
      y: fnrVals || [],
      mode: 'lines+markers',
      name: 'False Negative Rate (FNR) [Lower is Better]',
      line: { color: '#c084fc', width: 3, dash: 'dot' },
      marker: { size: 6, symbol: 'diamond' }
    };
    const traceFooling = {
      x: xVals,
      y: foolingVals || [],
      mode: 'lines+markers',
      name: 'Fooling Rate [Lower is Better]',
      line: { color: '#f472b6', width: 3, dash: 'dashdot' },
      marker: { size: 6, symbol: 'triangle-up' }
    };

    Plotly.newPlot(
      elId,
      [traceDR, traceFAR, traceABR, traceFNR, traceFooling],
      {
        title: 'IDS Resilience Security Curves (DR / FAR / ABR / FNR / Fooling Rate)',
        paper_bgcolor: '#142027',
        plot_bgcolor: '#142027',
        font: { color: '#e7f2f4', family: 'Manrope, sans-serif' },
        xaxis: {
          title: 'Adversarial Perturbation Strength (Epsilon Scale)',
          gridcolor: '#2b3f49',
          zerolinecolor: '#2b3f49'
        },
        yaxis: {
          title: 'Rate / Ratio (0.0 to 1.0)',
          gridcolor: '#2b3f49',
          zerolinecolor: '#2b3f49',
          range: [0, 1.05]
        },
        legend: {
          orientation: 'h',
          y: -0.25,
          x: 0.5,
          xanchor: 'center'
        },
        margin: { t: 60, l: 50, r: 20, b: 100 },
        hovermode: 'x unified'
      },
      { displayModeBar: false, responsive: true }
    );
  } catch(e) {
    console.error("Plotly error:", e);
    el.innerHTML = '<div style="padding:12px;color:var(--danger)">Chart rendering error.</div>';
  }
}

async function refreshMetrics(){
  const cardsEl = document.getElementById('metricCards');
  if(cardsEl) cardsEl.innerHTML = '<div class="card"><div class="k">Metrics</div><div class="v">Loading...</div></div>';
  const toggle2 = document.getElementById('globalDefenseToggle');
  const defenseActive = toggle2 ? toggle2.checked : true;
  const modelSel = document.getElementById('defActiveModelSelector');
  const activeModelSelected = modelSel ? modelSel.value : 'auto';
  
  const m = await api(`/metrics?defense_enabled=${defenseActive}&active_defense_model=${activeModelSelected}`);
  
  // Calculate differences to show trends
  const drDiff = m.dr_after - m.dr_before;
  const farDiff = m.far_after - m.far_before;
  const fnrDiff = m.fnr_after - m.fnr_before;
  
  const drTrend = drDiff < 0 
    ? `<span style="color:var(--danger); font-size:0.75rem;">📉 -${Math.abs(drDiff*100).toFixed(1)}% detection drop</span>`
    : `<span style="color:var(--good); font-size:0.75rem;">📈 +${Math.abs(drDiff*100).toFixed(1)}% detection gain</span>`;
    
  const farTrend = farDiff > 0 
    ? `<span style="color:var(--danger); font-size:0.75rem;">📈 +${Math.abs(farDiff*100).toFixed(1)}% false alarms</span>`
    : `<span style="color:var(--good); font-size:0.75rem;">📉 -${Math.abs(farDiff*100).toFixed(1)}% false alarms</span>`;

  const fnrTrend = fnrDiff > 0 
    ? `<span style="color:var(--danger); font-size:0.75rem;">📈 +${Math.abs(fnrDiff*100).toFixed(1)}% missed attacks</span>`
    : `<span style="color:var(--good); font-size:0.75rem;">📉 -${Math.abs(fnrDiff*100).toFixed(1)}% missed attacks</span>`;

  setCardsTone('metricCards',[
    {k:'Detection Rate (Clean)',v:`${(m.dr_before*100).toFixed(2)}%`,tone:'good'},
    {k:'Detection Rate (Under Attack)',v:`${(m.dr_after*100).toFixed(2)}% <br/>${drTrend}`,tone:m.dr_after < 0.75 ? 'bad' : 'warn'},
    {k:'False Alarm Rate (Clean)',v:`${(m.far_before*100).toFixed(2)}%`,tone:'info'},
    {k:'False Alarm Rate (Under Attack)',v:`${(m.far_after*100).toFixed(2)}% <br/>${farTrend}`,tone:m.far_after > 0.15 ? 'bad' : 'good'},
    {k:'FNR (Clean)',v:`${(m.fnr_before*100).toFixed(2)}%`,tone:m.fnr_before > 0.2 ? 'bad' : 'good'},
    {k:'FNR (Under Attack)',v:`${(m.fnr_after*100).toFixed(2)}% <br/>${fnrTrend}`,tone:m.fnr_after > 0.3 ? 'bad' : 'warn'},
    {k:'Fooling Rate',v:`${(m.fooling_rate*100).toFixed(2)}%`,tone:m.fooling_rate > 0.3 ? 'bad' : 'good'},
    {k:'Evasion Success Rate',v:`${(m.evasion_rate*100).toFixed(2)}%`,tone:m.evasion_rate > 0.4 ? 'bad' : 'good'},
    {k:'Traffic Flow Volume',v:`${m.sample_size ?? '-'} / ${m.sample_total ?? '-'}`,tone:'info'},
  ]);

  const curve = Array.isArray(m.curve) ? m.curve : [];
  const x = curve.map(r=>r.epsilon);
  const dr = curve.map(r=>r.detection_rate);
  const far = curve.map(r=>r.false_alarm_rate);
  const abr = curve.map(r=>1.0 - r.detection_rate);
  const fnr = curve.map(r=>r.fnr);
  const fooling = curve.map(r=>r.fooling_rate);
  
  renderCombinedMetricsChart(x, dr, far, abr, fnr, fooling);
  showToast('Intrusion metrics and security curves refreshed', 'ok', 1700);
}

async function runBenchmark(){
  const attackType = document.getElementById('benchAttackType').value;
  const epsilon = Number(document.getElementById('benchEpsilon').value);
  const sampleSize = Number(document.getElementById('benchSamples').value);
  const pgdSteps = Number(document.getElementById('benchPgdSteps').value);
  const useDefense = document.getElementById('benchDefense').checked;

  const attackTypes = attackType === 'all' ? ['uap','fgsm','pgd','random','query'] : [attackType];

  setCards('benchCards',[
    {k:'Benchmark',v:'Running...'}
  ]);

  const out = await api('/attack-benchmark','POST',{
    epsilon,
    sample_size: sampleSize,
    attack_types: attackTypes,
    use_defense: useDefense,
    pgd_steps: pgdSteps,
  });

  setCardsTone('benchCards',[
    {k:'Base Accuracy',v:`${(out.base_accuracy*100).toFixed(2)}%`,tone:'good'},
    {k:'Base FNR',v:`${(out.base_FNR*100).toFixed(2)}%`,tone:'warn'},
    {k:'Samples',v:String(out.sample_size),tone:'info'},
    {k:'Epsilon',v:out.epsilon.toFixed(3),tone:'warn'},
    {k:'Defense',v:out.use_defense?'ON':'OFF',tone:out.use_defense?'good':'bad'},
  ]);

  LAST_BENCHMARK = out;
  renderBenchCharts(out.results || [], !!out.use_defense);
  renderBenchTable(out.results || [], !!out.use_defense);
  showToast('Attack benchmark finished', 'ok');
}

async function refreshDrift(){
  const topK = Math.max(3, Math.min(30, Number(document.getElementById('driftTopK').value || 10)));
  const out = await api(`/drift?top_k=${encodeURIComponent(topK)}`);

  if(out.status !== 'ok'){
    setCardsTone('driftCards',[
      {k:'Drift Status',v:(out.status || 'unknown').toUpperCase(),tone:'warn'},
      {k:'Message',v:out.message || 'No live samples yet',tone:'info'},
    ]);
    renderDriftTable([]);
    return;
  }

  const top = Array.isArray(out.top_drift_features) ? out.top_drift_features : [];
  const meanScore = top.length ? (top.reduce((a,r)=>a + Number(r.drift_score || 0), 0) / top.length) : 0;

  setCardsTone('driftCards',[
    {k:'Reference Samples',v:String(out.reference_samples ?? '-'),tone:'info'},
    {k:'Live Samples',v:String(out.live_samples ?? '-'),tone:'info'},
    {k:'Top Features',v:String(top.length),tone:'warn'},
    {k:'Mean Drift Score',v:meanScore.toFixed(4),tone:meanScore>0.35?'bad':'good'},
  ]);
  renderDriftTable(top);
  if(meanScore > 0.35) showToast('High drift detected in live traffic', 'err');
}

function resetUi(){
  LAST_UAP = null;
  BEFORE_SAMPLE = null;
  setCards('defenseCards', []);
  setCards('attackCards', []);
  setCards('metricCards', []);
  setCards('driftCards', []);
  const t1 = document.getElementById('changesTable');
  const t2 = document.getElementById('driftTable');
  if(t1) t1.innerHTML = '';
  if(t2) t2.innerHTML = '';
  resetDemoTimeline();
  showToast('UI cards reset', 'warn', 1600);
}

async function runFullDemo(){
  resetDemoTimeline();
  setDemoTimelineState(1, 'Step 1/5: Loading attack sample');
  await loadSample('attack');

  await sleep(180);
  setDemoTimelineState(2, 'Step 2/5: Running defense prediction');
  await doPredict();

  await sleep(180);
  setDemoTimelineState(3, 'Step 3/5: Generating perturbation');
  await doGenerate();

  await sleep(180);
  setDemoTimelineState(4, 'Step 4/5: Applying attack');
  await doApply();

  await sleep(180);
  setDemoTimelineState(5, 'Step 5/5: Refreshing metrics');
  await refreshMetrics();

  await sleep(220);
  completeDemoTimeline();
  showToast('Full demo completed', 'ok');
}

function handleUiError(err){
  const msg = err && err.message ? err.message : 'Unexpected UI error';
  showToast(msg, 'err', 3200);
}

let pollTimer = null;
function pollTrainingStatus(){
  if(pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const status = await api('/train-status');
      const stateLabel = document.getElementById('trainingStateLabel');
      const logsArea = document.getElementById('trainingLogs');
      const pulse = document.getElementById('trainingPulse');
      const container = document.getElementById('trainingStatusContainer');
      
      if(container) container.style.display = 'block';
      if(stateLabel) stateLabel.textContent = `Status: ${status.status.toUpperCase()}`;
      if(logsArea) {
        logsArea.value = status.logs;
        logsArea.scrollTop = logsArea.scrollHeight;
      }
      if(pulse) {
        pulse.className = status.status === 'running' ? 'pulseDot' : '';
        pulse.style.background = status.status === 'running' ? 'var(--warning)' : (status.status === 'finished' ? 'var(--good)' : 'var(--danger)');
      }
      
      if(status.status !== 'running') {
        clearInterval(pollTimer);
        pollTimer = null;
        const btn = document.getElementById('btnTrainModel');
        if(btn) btn.disabled = false;
        showToast(`Training ${status.status}`, status.status === 'finished' ? 'ok' : 'err');
        await refreshBackendHealth();
        if(BEFORE_SAMPLE) doPredict();
      }
    } catch(e) {
      console.warn("Polling error:", e);
    }
  }, 1000);
}

async function triggerModelTraining(){
  const modelName = document.getElementById('trainModelSelector').value;
  const steps = Number(document.getElementById('trainSteps').value || 2000);
  const ratio = Number(document.getElementById('trainAdvRatio').value || 0.5);
  
  const btn = document.getElementById('btnTrainModel');
  if(btn) btn.disabled = true;
  
  const container = document.getElementById('trainingStatusContainer');
  if(container) container.style.display = 'block';
  
  try {
    const res = await api('/train-model', 'POST', {
      model_name: modelName,
      timesteps: steps,
      adv_ratio: ratio
    });
    showToast(res.message, 'warn');
    pollTrainingStatus();
  } catch(e) {
    if(btn) btn.disabled = false;
    handleUiError(e);
  }
}

window.addEventListener('DOMContentLoaded', async ()=>{
  updateActivityStrip();
  document.getElementById('epsilon').addEventListener('input',e=>{document.getElementById('epsVal').textContent=Number(e.target.value).toFixed(3)});
  
  // Global Defense Toggle listener (Synchronized across Topbar and Section 1)
  const toggle1 = document.getElementById('globalDefenseToggleTopbar');
  const text1 = document.getElementById('globalDefenseStatusTextTopbar');
  const toggle2 = document.getElementById('globalDefenseToggle');
  const text2 = document.getElementById('globalDefenseStatusText');

  function syncDefenseState(active) {
    if (toggle1) toggle1.checked = active;
    if (toggle2) toggle2.checked = active;
    const stateStr = active ? 'ON' : 'OFF';
    const stateColor = active ? 'var(--primary-2)' : 'var(--danger)';
    if (text1) {
      text1.textContent = stateStr;
      text1.style.color = stateColor;
    }
    if (text2) {
      text2.textContent = stateStr;
      text2.style.color = stateColor;
    }
  }

  if (toggle1) {
    toggle1.addEventListener('change', () => {
      const active = toggle1.checked;
      syncDefenseState(active);
      showToast(active ? 'Defense Shield ACTIVE' : 'Defense Shield DEACTIVATED', active ? 'ok' : 'err');
      if(BEFORE_SAMPLE) doPredict();
      refreshMetrics().catch(()=>{});
    });
  }
  if (toggle2) {
    toggle2.addEventListener('change', () => {
      const active = toggle2.checked;
      syncDefenseState(active);
      showToast(active ? 'Defense Shield ACTIVE' : 'Defense Shield DEACTIVATED', active ? 'ok' : 'err');
      if(BEFORE_SAMPLE) doPredict();
      refreshMetrics().catch(()=>{});
    });
  }

  const modelSel = document.getElementById('defActiveModelSelector');
  if (modelSel) {
    modelSel.addEventListener('change', () => {
      const active = toggle2 ? toggle2.checked : true;
      if (LAST_APPLY_RESULT) {
        updateDefenseDashboard(LAST_APPLY_RESULT, LAST_ATTACK_TYPE, active);
      }
      refreshMetrics().catch(()=>{});
    });
  }

  const chkOnlyPerturbed = document.getElementById('chkOnlyPerturbed');
  if (chkOnlyPerturbed) {
    chkOnlyPerturbed.addEventListener('change', () => {
      if (BEFORE_SAMPLE && LAST_APPLY_RESULT) {
        renderChanges(LAST_APPLY_RESULT.modified_input, BEFORE_SAMPLE);
      }
    });
  }

  // Training lab buttons
  document.getElementById('btnTrainModel').addEventListener('click', triggerModelTraining);

  // Attack simulator buttons
  document.getElementById('btnStartSimulation').addEventListener('click', () => startAttackSimulation().catch(handleUiError));
  document.getElementById('btnStopSimulation').addEventListener('click', () => stopAttackSimulation());

  const btnSingleAttack = document.getElementById('btnSingleAttack');
  if (btnSingleAttack) {
    btnSingleAttack.addEventListener('click', () => triggerSingleAttack().catch(handleUiError));
  }

  const btnClearHistory = document.getElementById('btnClearAttackHistory');
  if (btnClearHistory) {
    btnClearHistory.addEventListener('click', clearAttackHistory);
  }

  const btnExportCsv = document.getElementById('btnExportAttackCsv');
  if (btnExportCsv) {
    btnExportCsv.addEventListener('click', exportAttackHistoryCsv);
  }

  const btnExportJson = document.getElementById('btnExportAttackJson');
  if (btnExportJson) {
    btnExportJson.addEventListener('click', exportAttackHistoryJson);
  }

  const btnExportTxt = document.getElementById('btnExportAttackTxt');
  if (btnExportTxt) {
    btnExportTxt.addEventListener('click', exportAttackHistoryTxt);
  }

  document.getElementById('btnRefreshMetrics').addEventListener('click',()=>withLoading('btnRefreshMetrics', ()=>refreshMetrics()).catch(handleUiError));
  const btnDemo = document.getElementById('btnDemoTimelineRun');
  if (btnDemo) {
    btnDemo.addEventListener('click',()=>withLoading('btnDemoTimelineRun', ()=>runFullDemo()).catch(handleUiError));
  }
 
  try{
    initAttackHistory();
    await refreshBackendHealth();
    
    // Resume polling training status if running on load
    const initialStatus = await api('/train-status');
    if(initialStatus.status === 'running') {
      const btn = document.getElementById('btnTrainModel');
      if(btn) btn.disabled = true;
      pollTrainingStatus();
    }

    await loadSample('attack');
    await doPredict();
    await refreshMetrics();
    resetDemoTimeline();
    setInterval(()=>{refreshBackendHealth().catch(()=>{})}, 15000);
  }catch(e){
    console.warn(e.message);
    handleUiError(e);
  }
});
