# adversarial-attack-on-IDS
Source code for the paper 'Generative Adversarial Attacks Against Intrusion Detection Systems Using Active Learning'.

The code is written and tested in Python 3.6 and PyTorch 1.2.0.

Requires the network traffic flow data file 'MachineLearningCSV.zip' from the CICIDS 2017 dataset (available for free at https://www.unb.ca/cic/datasets/ids-2017.html).

To generate the adversarial attack using the Gen-AAL algorithm:
1. Run 'data_preprocessing.py' to generate the data files 'NBx.npy' and 'NBy.npy'.
2. Run 'pretrain_vae.py' to train the IDS model (Optional).
3. Run 'pretrain_vae.py' to initialize the VAE model (Optional).
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

- `http://127.0.0.1:8000/`

This UI lets you simulate both:

- `Defense`: clean prediction + alert generation
- `Attack`: UAP-based perturbation and before/after decision comparison

Available endpoints:

- `GET /health`: service status and feature shape
- `GET /schema`: expected feature keys (`f0 ... fN`)
- `POST /predict`: predict one sample and generate alert if intrusion confidence is high
- `POST /predict_batch`: predict many samples at once
- `POST /predict_csv`: upload flow CSV and get batch attack stats
- `GET /simulation_sample`: fetch a sample from `NBx.npy` for UI simulation
- `POST /simulate_attack`: run clean vs adversarial (UAP) inference
- `GET /alerts?limit=100`: recent in-memory alerts

Single prediction request example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
	-H "Content-Type: application/json" \
	-d "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}"
```

Alerts are also persisted to JSONL at:

- `logs/ids_alerts.jsonl`

## FastAPI Interactive Defense + Attack System

This repository now also includes a fully interactive real-time system with two connected panels:

- Defense Panel (IDS)
- Attack Panel (Adversarial Engine)

Run:

```bash
python fastapi_ids_system.py
```

Open:

- `http://127.0.0.1:9000/`

Core FastAPI endpoints:

- `POST /predict`: run IDS prediction on input sample
- `POST /generate-attack`: generate/update UAP using FGSM logic and constraints
- `POST /apply-attack`: apply UAP, enforce constraints, re-predict and detect label change
- `GET /metrics`: compute before/after metrics and epsilon curves

Additional helper endpoint:

- `GET /sample?kind=attack|benign`: load test sample for interactive runs
