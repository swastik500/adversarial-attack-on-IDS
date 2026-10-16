# adversarial-attack-on-IDS
Source code for the paper 'Generative Adversarial Attacks Against Intrusion Detection Systems Using Active Learning'.

The repository now includes both the original paper pipeline and a newer forward pipeline
(DRL IDS + UAP + PCC + interactive APIs/UI). Use Python 3.10+ for best compatibility with
`stable-baselines3`, `fastapi`, and modern scientific Python packages.

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements_forward.txt
```

Requires the network traffic flow data file 'MachineLearningCSV.zip' from the CICIDS 2017 dataset (available for free at https://www.unb.ca/cic/datasets/ids-2017.html).

To generate the adversarial attack using the Gen-AAL algorithm:
1. Run 'data_preprocessing.py' to generate the data files 'NBx.npy' and 'NBy.npy'.
2. Run 'train_ids_gbt.py' to train the baseline GBT IDS model (Optional, for legacy scripts).
3. Run 'pretrain_vae.py' to initialize/train the VAE model (Optional).
4. Run 'attack_Gen-AAL.py' to generate adversarial attacks using the Gen-AAL algorithm and see the attack success rate.
5. For performance comparison, run 'attack_DFAL.py' to generate adversarial attacks using the DFAL algorithm.

The Gen-AAL algorithm will be improved in the future updates.

## Forward Project Pipeline (DRL IDS + UAP + PCC)

This repository now includes an end-to-end script for the forward phase:

1. Load preprocessed CICIDS feature arrays (`NBx.npy`, `NBy.npy`)
2. Train a DRL IDS classifier with DQN (`stable-baselines3`)
3. Generate attacks with FGSM or BIM
4. Build a UAP under feature constraints (MF/RF/UF)
5. Refine perturbation with PCC-based objective
6. Evaluate `Accuracy`, `FNR`, and `Fooling Rate`

Run:

```bash
python forward_project.py --timesteps 10000 --attack fgsm --uap-epochs 3
```

Quick smoke test:

```bash
python forward_project.py --timesteps 1000 --attack fgsm --uap-epochs 1
```

Optional feature groups (comma-separated indices):

```bash
python forward_project.py --mf 0,1,2,3,4 --rf 5,6,7,8 --uf 9,10,11
```

Saved outputs:

- `trained_models/dqn_ids_model.zip`
- `trained_models/uap.npy`
- `trained_models/forward_metrics.json`

When run-metrics saving is enabled (default), each run also writes a timestamped file to:

- `trained_models/forward_runs/metrics_*.json`

## Threat Model

This project currently assumes an evasion-focused white/gray-box attacker with the following scope:

- Attacker knowledge:
	- Understands feature schema (`f0...fN`) and normalization range `[0,1]`.
	- Can approximate model behavior through repeated queries and confidence outputs.
	- May know the broad classifier family (DQN IDS), but not necessarily exact weights.
- Perturbation budget:
	- Bounded by $\epsilon$ in L-infinity style constraints (e.g., `--epsilon 0.03`).
	- Per-sample perturbations are clipped to keep features in `[0,1]`.
- Constrained features:
	- `MF` (modifiable features): directly perturbed.
	- `UF` (unmodifiable features): explicitly preserved.
	- `RF` (recomputable features): recalculated from constrained relations.

You can set custom constraints with:

```bash
python forward_project.py --mf 0,1,2 --rf 3,4 --uf 5,6,7
```

## Statistical Significance Reporting

Run multiple seeds to estimate robustness with mean/std and 95% CI:

```bash
python forward_project.py --seed 1 --timesteps 3000 --uap-epochs 2
python forward_project.py --seed 2 --timesteps 3000 --uap-epochs 2
python forward_project.py --seed 3 --timesteps 3000 --uap-epochs 2
```

Aggregate repeated runs:

```bash
python report_significance.py --runs-dir trained_models/forward_runs --min-runs 3
```

Saved summary:

- `trained_models/forward_significance.json`

## Web Dashboard

An interactive HTML/CSS dashboard is available to visualize IDS behavior before/after UAP attack:

```bash
python web_dashboard_server.py
```

The dashboard includes:

- Dataset panel (size, features, class distribution, sample rows)
- IDS prediction panel (label + confidence)
- UAP attack simulation with changed feature highlighting
- Before/after comparison panel
- Metrics cards (Accuracy, FNR, Fooling Rate)
- Epsilon curves (Accuracy, FNR, Fooling Rate)
- Confusion matrices before and after attack
- Feature impact visualization
- End-to-end workflow diagram

## Real-World IDS Service (API)

To run this as an operational IDS inference service (not only visualization), start:

```bash
python real_world_ids_server.py --host 0.0.0.0 --port 8000
```

Open in browser:

- `http://127.0.0.1:8000/` — Full-featured operations console

**Service UI Features (Now Fully Implemented)**:

1. **Service Status Dashboard** — Real-time health metrics
   - Service status, feature count, alert threshold
   - UAP and simulation dataset availability
   - Live sample counter, auto-refresh every 30 seconds

2. **Defense Panel** — Single prediction & threat detection
   - Load samples by index (auto-random if not specified)
   - Paste normalized features [0, 1]
   - Run prediction with confidence and alert status
   - Timestamp logging for audit trails

3. **Attack Panel** — Adversarial UAP simulation
   - Configurable epsilon (perturbation budget: 0.001–0.1)
   - Before/after label and confidence comparison
   - Decision change detection
   - Top 15 perturbed features visualization

4. **Batch Prediction** — CSV upload & bulk processing
   - Upload CSV with f0...f77 columns (or any 78 numeric columns)
   - Batch attack detection with statistics
   - Results table: sample indices, predictions, confidence

5. **Drift Monitoring** — Live distribution shift detection
   - Population Stability Index (PSI) + Z-score per feature
   - Top 15 drifted features ranked by drift_score
   - Reference vs. live sample statistics
   - Real-time status: unavailable → insufficient → ok

6. **Alert Management** — Real-time threat logging
   - Auto-alert when prediction=Attack and confidence ≥ threshold
   - Recent alerts table (up to 500 in-memory, persisted to JSONL)
   - Alert severity (high/medium) based on confidence

7. **Toast Notifications** — User feedback
   - Success (✓), error (🚨), info (ℹ) toasts
   - Auto-dismiss or manual close
   - Position: top-right corner

8. **Loading States** — Disabled buttons during async ops
   - Visual feedback during API calls
   - Prevents double-clicks

Available endpoints (all integrated into UI):

- `GET /health`: service status and uptime
- `GET /schema`: feature schema and capability info
- `POST /predict`: single prediction with alert logic
- `POST /predict_batch`: batch predictions from items list
- `POST /predict_csv`: batch predictions from multipart CSV
- `GET /simulation_sample`: load a sample from NBx.npy
- `POST /simulate_attack`: UAP attack with epsilon control
- `GET /drift?top_k=15`: top drifted features and distribution shifts
- `GET /alerts?limit=50`: recent alerts from in-memory log

Single prediction request example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}"
```

Alerts are also persisted to JSONL at:

- `logs/ids_alerts.jsonl`

Example drift check:

```bash
curl "http://127.0.0.1:8000/drift?top_k=10"
```

## FastAPI Interactive Defense + Attack System

This repository now also includes a fully interactive real-time system with a modern landing page and connected defense/attack panels:

Run:

```bash
python fastapi_ids_system.py
```

Open in browser:

- **Landing Page**: `http://127.0.0.1:9001/` — Modern animated entry point with project overview
- **Interactive Dashboard**: `http://127.0.0.1:9001/dashboard` — Full defense + attack panels, metrics, and drift monitoring

Core FastAPI endpoints:

- `GET /health`: service status, uptime, device, and model shape
- `POST /predict`: run IDS prediction on input sample
- `POST /generate-attack`: generate/update UAP using FGSM logic and constraints
- `POST /apply-attack`: apply UAP, enforce constraints, re-predict and detect label change
- `GET /metrics`: compute before/after metrics and epsilon curves
- `POST /attack-benchmark`: benchmark UAP/FGSM/PGD/Random attacks with optional feature-squeeze defense

Additional helper endpoint:

- `GET /sample?kind=attack|benign`: load test sample for interactive runs

## Quick Troubleshooting

- FastAPI server with landing page: Run `python fastapi_ids_system.py` and go to `http://127.0.0.1:9001/`. You'll see the landing page first, then click "Launch FastAPI Dashboard" or navigate to `http://127.0.0.1:9001/dashboard` for the interactive UI.
- If you see errors like `Import "stable_baselines3" could not be resolved`, install dependencies from `requirements_forward.txt` in the active virtual environment.
- If FastAPI UI does not open, confirm the running port matches the launch command (default in code is `9001`).
- Ensure artifacts exist before running services:
	- `trained_models/dqn_ids_model.zip`
	- `trained_models/uap.npy`
