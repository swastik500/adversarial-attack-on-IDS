# Project Modules

This file lists the actual modules in this repository and keeps the same table-style format without unrelated workbook content.

## Module Plan

| Sr. No. | Module | Problem | Solution | Tech Stack | API Requirements |
| --- | --- | --- | --- | --- | --- |
| 1 | `data_preprocessing.py` | Raw IDS data must be cleaned and converted into model-ready arrays | Loads CICIDS data, cleans it, and saves `NBx.npy` / `NBy.npy` for training and inference | Python, NumPy, Pandas, scikit-learn | Input: CICIDS CSV files. Output: `NBx.npy`, `NBy.npy` |
| 2 | `train_ids_gbt.py` | The legacy pipeline needs a baseline IDS classifier | Trains the gradient-boosted IDS model used by older scripts | Python, scikit-learn, pickle | Consumes `NBx.npy` / `NBy.npy` and writes the trained GBT model |
| 3 | `pretrain_vae.py` | The older attack pipeline needs a generative model for feature reconstruction | Pretrains the VAE used in adversarial attack generation | Python, PyTorch, NumPy | Consumes `NBx.npy` / `NBy.npy`; writes VAE weights |
| 4 | `attack_Gen-AAL.py` | Need to generate adversarial examples for the Gen-AAL workflow | Builds adversarial samples and reports attack success rate | Python, PyTorch, NumPy | Uses trained IDS/VAE artifacts and preprocessed arrays |
| 5 | `attack_DFAL.py` | Need a comparison attack baseline for evaluation | Generates DFAL adversarial examples for comparison | Python, PyTorch, NumPy | Uses the same local model/data artifacts as the Gen-AAL pipeline |
| 6 | `forward_project.py` | The forward pipeline needs DRL IDS training, UAP generation, and constrained attack evaluation | Trains DQN IDS, generates UAPs, applies MF/RF/UF constraints, and evaluates Accuracy/FNR/Fooling Rate | Python, stable-baselines3, PyTorch, NumPy | Reads `NBx.npy` / `NBy.npy`; writes `trained_models/dqn_ids_model.zip`, `trained_models/uap.npy`, `trained_models/forward_metrics.json` |
| 7 | `report_significance.py` | Repeated runs need a consolidated robustness report | Aggregates multiple runs and computes mean/std/CI summaries | Python, NumPy, JSON | Reads `trained_models/forward_runs/*.json` |
| 8 | `fastapi_ids_system.py` | The project needs an interactive defense-and-attack API | Serves prediction, attack generation, attack application, metrics, and benchmark endpoints | Python, FastAPI, PyTorch, stable-baselines3 | Reads `trained_models/dqn_ids_model.zip`, `trained_models/uap.npy`, and sample arrays |
| 9 | `real_world_ids_server.py` | The project needs an operational IDS inference service | Serves health, schema, predict, batch, CSV, drift, alerts, and attack simulation endpoints | Python, Flask, PyTorch, stable-baselines3 | Uses `trained_models/uap.npy`, `NBx.npy`, and alert logs |
| 10 | `web_dashboard_server.py` | The project needs a browser dashboard for attack visualization | Serves the dashboard UI and runs the clean-vs-attacked comparison workflow | Python, Flask, PyTorch, Plotly, HTML/CSS | Reads `NBx.npy`, `NBy.npy`, `trained_models/dqn_ids_model.zip`, and `trained_models/uap.npy` |
| 11 | `dashboard_app.py` | The project needs a Streamlit-based visualization app | Provides a local dashboard for dataset inspection, prediction, and attack simulation | Python, Streamlit, PyTorch, Plotly | Reads `NBx.npy`, `NBy.npy`, `trained_models/dqn_ids_model.zip`, and `trained_models/uap.npy` |
| 12 | `models.py` | Shared neural-network definitions are needed across the pipeline | Defines the IDS model and VAE architectures | Python, PyTorch | Imported by training and attack scripts |
| 13 | `utils.py` | Shared dataset utilities are needed across training and evaluation scripts | Creates train/test splits and wraps IDS data in a PyTorch dataset | Python, NumPy, PyTorch, scikit-learn | Reads `NBx.npy` / `NBy.npy` and optional GBT artifacts |

## Notes

- This list is based on the repository files and the documented pipeline in the README and architecture diagrams.
- I left out datasets, model artifacts, and UI static files so the table stays focused on actual code modules.
