# Architecture Module Breakdown

This document divides the repository into major functional modules for a university mini/major project report. The grouping is architecture-level, not file-level.

## 1. Data Preparation and Feature Engineering

| Field | Details |
| --- | --- |
| Module Name | Data Preparation and Feature Engineering |
| Objective | Convert raw IDS traffic data into normalized train/test arrays that can be consumed by the rest of the system. |
| Problem Statement | Raw CICIDS-style data must be cleaned, standardized, split, and stored in a reusable format before model training or inference. |
| Proposed Solution | Load the source CSV data, remove invalid rows, normalize the feature space, and save the processed arrays as NBx.npy and NBy.npy. Provide reusable splitting and dataset wrappers for downstream scripts. |
| Files Included | data_preprocessing.py, utils.py, README.md |
| Technologies Used | Python, NumPy, Pandas, scikit-learn, PyTorch |
| APIs Used | NumPy file I/O, pandas CSV handling, scikit-learn train/test splitting, PyTorch Dataset utilities |
| Inputs | Raw CICIDS dataset files, especially the source CSV data referenced in the README. |
| Outputs | NBx.npy, NBy.npy, train/test splits, PyTorch dataset objects. |

## 2. Legacy IDS Training and Adversarial Research Pipeline

| Field | Details |
| --- | --- |
| Module Name | Legacy IDS Training and Adversarial Research Pipeline |
| Objective | Train the earlier IDS baseline and generate adversarial examples for research comparison. |
| Problem Statement | The original paper pipeline needs a baseline classifier, a generative model, and attack scripts to evaluate attack success and compare methods. |
| Proposed Solution | Train the gradient-boosted IDS baseline, pretrain the VAE, and use the Gen-AAL and DFAL attack scripts to generate adversarial samples and measure attack success. |
| Files Included | train_ids_gbt.py, pretrain_vae.py, attack_Gen-AAL.py, attack_DFAL.py, models.py |
| Technologies Used | Python, PyTorch, scikit-learn, NumPy, pickle, gradient boosting, VAE architecture |
| APIs Used | PyTorch neural-network layers, scikit-learn model training utilities, pickle serialization |
| Inputs | NBx.npy, NBy.npy, trained baseline artifacts, VAE weights, and pretrained model files. |
| Outputs | Trained GBT IDS model, VAE weights, adversarial samples, attack success metrics, comparison results. |

## 3. Forward DRL IDS and UAP Generation Pipeline

| Field | Details |
| --- | --- |
| Module Name | Forward DRL IDS and UAP Generation Pipeline |
| Objective | Build the modern end-to-end attack pipeline that trains a DRL-based IDS, generates a universal adversarial perturbation, and evaluates the attack under feature constraints. |
| Problem Statement | The project needs a forward pipeline that goes beyond the legacy attack scripts and supports constrained attack generation, metric evaluation, and artifact persistence. |
| Proposed Solution | Load the preprocessed arrays, train a DQN IDS classifier, generate FGSM or BIM attacks, optimize a UAP under MF/RF/UF constraints, refine it with a PCC objective, and save model, perturbation, and metric artifacts. |
| Files Included | forward_project.py, PLANTUML_DIAGRAMS.md, README.md |
| Technologies Used | Python, stable-baselines3, PyTorch, NumPy, scikit-learn, Gymnasium, JSON |
| APIs Used | DQN from stable-baselines3, PyTorch tensor operations, sklearn train/test split and metrics, JSON artifact persistence |
| Inputs | NBx.npy, NBy.npy, feature constraint indices, attack configuration parameters, seed values. |
| Outputs | trained_models/dqn_ids_model.zip, trained_models/uap.npy, trained_models/forward_metrics.json, trained_models/forward_runs/*.json. |

## 4. Operational IDS API Service

| Field | Details |
| --- | --- |
| Module Name | Operational IDS API Service |
| Objective | Provide a real-time backend for single prediction, batch prediction, CSV upload, drift monitoring, alert handling, and attack simulation. |
| Problem Statement | The IDS needs to be usable as an operational service, not only as an offline training pipeline. |
| Proposed Solution | Serve the trained DQN model through web APIs, load the UAP artifact, support prediction and attack endpoints, track alerts, and expose drift and batch-processing functions. |
| Files Included | fastapi_ids_system.py, real_world_ids_server.py, trained_models/uap.npy, trained_models/dqn_ids_model.zip, logs/ids_alerts.jsonl |
| Technologies Used | Python, FastAPI, Flask, PyTorch, stable-baselines3, NumPy, Pydantic, pandas |
| APIs Used | GET /health, GET /schema, POST /predict, POST /predict_batch, POST /predict_csv, GET /simulation_sample, POST /simulate_attack, GET /drift, GET /alerts, POST /generate-attack, POST /apply-attack, GET /metrics, POST /attack-benchmark |
| Inputs | Feature vectors, CSV uploads, sample indices, epsilon values, NBx.npy, NBy.npy, trained models, UAP artifacts. |
| Outputs | Prediction responses, attack simulations, drift summaries, alerts, metrics, benchmark results, JSON responses for UI clients. |

## 5. Visualization and Interactive Dashboard Layer

| Field | Details |
| --- | --- |
| Module Name | Visualization and Interactive Dashboard Layer |
| Objective | Present the IDS workflow and attack effects in browser-based and local visual dashboards for analysis and demonstration. |
| Problem Statement | The project needs clear visual interfaces to inspect predictions, compare clean versus attacked samples, and communicate results to users. |
| Proposed Solution | Provide a web dashboard, Streamlit-based app, and static UI pages that render metrics, dataset summaries, attack comparisons, and workflow diagrams. |
| Files Included | web_dashboard_server.py, dashboard_app.py, web/index.html, web/app.js, web/styles.css, fastapi_ui/index.html, fastapi_ui/app.js, fastapi_ui/styles.css, service_ui/index.html, service_ui/app.js, service_ui/styles.css, index.html |
| Technologies Used | Python, Flask, Streamlit, HTML, CSS, JavaScript, Plotly |
| APIs Used | Flask routes for dashboard delivery, Streamlit rendering, Plotly charting, browser-side fetch calls to backend endpoints |
| Inputs | Model predictions, UAP outputs, evaluation metrics, sample rows, dataset statistics, attack simulation responses. |
| Outputs | Dashboard pages, charts, confusion matrices, before/after comparison panels, feature impact visualizations. |

## 6. Shared Modeling and Support Layer

| Field | Details |
| --- | --- |
| Module Name | Shared Modeling and Support Layer |
| Objective | Centralize neural-network definitions and reusable helpers that are shared across training, attack, and serving modules. |
| Problem Statement | Multiple scripts need the same model definitions, feature constraints, dataset wrappers, and loading logic, which should not be duplicated conceptually in the architecture. |
| Proposed Solution | Keep common neural-network classes, constraint logic, and dataset helpers in shared support modules that are imported by the main pipelines and services. |
| Files Included | models.py, utils.py, forward_project.py, fastapi_ids_system.py, real_world_ids_server.py, web_dashboard_server.py |
| Technologies Used | Python, PyTorch, NumPy, scikit-learn, dataclasses, typing |
| APIs Used | PyTorch nn.Module, dataset utilities, standard Python data structures, NumPy array operations |
| Inputs | Feature arrays, labels, model checkpoints, prediction inputs, and configuration parameters from other modules. |
| Outputs | Shared model objects, dataset wrappers, reusable feature-constraint behavior, helper transformations, and prediction utilities. |

## Summary

The repository can be presented as a six-module university project: data preparation, legacy adversarial research, forward DRL IDS pipeline, operational API service, visualization/dashboard layer, and shared support components. This grouping avoids one-file-per-module and reflects the system architecture shown in the README and PlantUML diagrams.