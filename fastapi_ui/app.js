let LAST_UAP = null;
let BEFORE_SAMPLE = null;

function card(k,v){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`}
function setCards(id,items){document.getElementById(id).innerHTML=items.map(x=>card(x.k,x.v)).join('')}

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
  const r=await fetch(url,opts);
  const j=await r.json();
  if(!r.ok) throw new Error(j.detail||j.error||'Request failed');
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
      [{x:xVals,y:yVals,mode:'lines+markers',name:title,line:{color:'#00f0ff'}}],
      {
        title,
        paper_bgcolor:'#1d2023',
        plot_bgcolor:'#1d2023',
        font:{color:'#e1e2e7'},
        xaxis:{gridcolor:'#2b3436'},
        yaxis:{gridcolor:'#2b3436'},
        margin:{t:40,l:40,r:20,b:40}
      },
      {displayModeBar:false,responsive:true}
    );
  }catch(_){
    renderChartFallback(elId, xVals, yVals, title);
  }
}

function renderChanges(mod,orig){
  const t=document.getElementById('changesTable');
  if(!mod||!orig){t.innerHTML='';return}
  const rows=[];
  for(let i=0;i<Math.min(mod.length,orig.length);i++){
    const d=mod[i]-orig[i];
    rows.push({feature:`f${i}`,before:orig[i],after:mod[i],delta:d,changed:Math.abs(d)>1e-9});
  }
  const cols=['feature','before','after','delta'];
  const th=`<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
  const tb=rows.map(r=>`<tr>${cols.map(c=>`<td class="${r.changed?'changed':''}">${typeof r[c]==='number'?r[c].toFixed(6):r[c]}</td>`).join('')}</tr>`).join('');
  t.innerHTML=`<thead>${th}</thead><tbody>${tb}</tbody>`;
}

async function loadSample(kind){
  const j=await api(`/sample?kind=${encodeURIComponent(kind)}`);
  document.getElementById('sampleInput').value=j.sample.join(', ');
}

async function doPredict(){
  const s=parseSample();
  BEFORE_SAMPLE=s;
  const j=await api('/predict','POST',{sample:s});
  setCards('defenseCards',[
    {k:'Prediction',v:j.label.toUpperCase()},
    {k:'Confidence',v:`${(j.confidence*100).toFixed(2)}%`}
  ]);
}

async function doGenerate(){
  const eps=Number(document.getElementById('epsilon').value);
  const j=await api('/generate-attack','POST',{epsilon:eps});
  LAST_UAP=j.uap;
  setCards('attackCards',[
    {k:'UAP Length',v:j.uap.length},
    {k:'Epsilon',v:j.epsilon.toFixed(4)}
  ]);
}

async function doApply(){
  const s=parseSample();
  if(!BEFORE_SAMPLE) BEFORE_SAMPLE=[...s];
  const body={sample:s};
  if(LAST_UAP) body.uap=LAST_UAP;
  const j=await api('/apply-attack','POST',body);

  setCards('attackCards',[
    {k:'Before',v:j.original_prediction.label.toUpperCase()},
    {k:'After',v:j.new_prediction.label.toUpperCase()},
    {k:'Changed',v:j.prediction_changed?'YES':'NO'},
    {k:'Used Scale',v:j.used_scale.toFixed(2)}
  ]);

  renderChanges(j.modified_input, s);
}

async function refreshMetrics(){
  const cardsEl = document.getElementById('metricCards');
  cardsEl.innerHTML = '<div class="card"><div class="k">Metrics</div><div class="v">Loading...</div></div>';
  renderChartFallback('accChart', [], [], 'Accuracy vs Epsilon');
  renderChartFallback('fnrChart', [], [], 'FNR vs Epsilon');
  renderChartFallback('foolChart', [], [], 'Fooling Rate vs Epsilon');

  const m=await api('/metrics');
  setCards('metricCards',[
    {k:'Accuracy Before',v:`${(m.accuracy_before*100).toFixed(2)}%`},
    {k:'Accuracy After',v:`${(m.accuracy_after*100).toFixed(2)}%`},
    {k:'FNR Before',v:`${(m.FNR_before*100).toFixed(2)}%`},
    {k:'FNR After',v:`${(m.FNR_after*100).toFixed(2)}%`},
    {k:'Fooling Rate',v:`${(m.fooling_rate*100).toFixed(2)}%`},
    {k:'Eval Samples',v:`${m.sample_size ?? '-'} / ${m.sample_total ?? '-'}`},
  ]);

  const curve = Array.isArray(m.curve) ? m.curve : [];
  const x=curve.map(r=>r.epsilon);
  renderMetricChart('accChart', x, curve.map(r=>r.accuracy), 'Accuracy vs Epsilon');
  renderMetricChart('fnrChart', x, curve.map(r=>r.fnr), 'FNR vs Epsilon');
  renderMetricChart('foolChart', x, curve.map(r=>r.fooling_rate), 'Fooling Rate vs Epsilon');
}

window.addEventListener('DOMContentLoaded', async ()=>{
  document.getElementById('epsilon').addEventListener('input',e=>{document.getElementById('epsVal').textContent=Number(e.target.value).toFixed(3)});
  document.getElementById('btnLoadAttack').addEventListener('click',()=>loadSample('attack').catch(e=>alert(e.message)));
  document.getElementById('btnLoadBenign').addEventListener('click',()=>loadSample('benign').catch(e=>alert(e.message)));
  document.getElementById('btnPredict').addEventListener('click',()=>doPredict().catch(e=>alert(e.message)));
  document.getElementById('btnGenerate').addEventListener('click',()=>doGenerate().catch(e=>alert(e.message)));
  document.getElementById('btnApply').addEventListener('click',()=>doApply().catch(e=>alert(e.message)));
  document.getElementById('btnRefreshMetrics').addEventListener('click',()=>refreshMetrics().catch(e=>alert(e.message)));

  try{
    await loadSample('attack');
    await doPredict();
    await refreshMetrics();
  }catch(e){console.warn(e.message)}
});
