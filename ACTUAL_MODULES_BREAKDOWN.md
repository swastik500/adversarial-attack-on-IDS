# Actual Module Breakdown

This document describes the repository as seven functional modules based on the actual system behavior in the codebase. The grouping is architecture-level and suitable for a university project report.

## 1) Analyze Input (Defense)

| Field | Details |
| --- | --- |
| Module Name | Analyze Input (Defense) |
| Objective | Accept a network sample, validate it, and produce a clean IDS prediction with confidence. |
| Problem Statement | The system must safely analyze raw or normalized input features before any attack simulation is applied. |
| Proposed Solution | Load the trained IDS model, coerce the input into a fixed feature vector, clamp values to the valid range, and return the prediction and confidence. |
| Files Included | fastapi_ids_system.py, real_world_ids_server.py, dashboard_app.py, web_dashboard_server.py, utils.py |
| Technologies Used | Python, FastAPI, Flask, Streamlit, PyTorch, NumPy, pandas |
| APIs Used | POST /predict, GET /health, sample-loading helpers, prediction utilities |
| Inputs | Feature vectors, sample indices, CSV rows, feature dictionaries keyed as f0..fN. |
| Outputs | Label prediction, confidence score, alert status, prediction JSON, dashboard display values. |

## 2) Simulate Attack

| Field | Details |
| --- | --- |
| Module Name | Simulate Attack |
| Objective | Generate and apply adversarial perturbations to test how the IDS reacts under attack. |
| Problem Statement | The project needs a controlled way to attack the IDS and observe whether the prediction changes. |
| Proposed Solution | Build or load a UAP, apply FGSM/BIM-style perturbations under epsilon constraints, and compare clean versus attacked predictions. |
| Files Included | forward_project.py, fastapi_ids_system.py, real_world_ids_server.py, web_dashboard_server.py, attack_Gen-AAL.py, attack_DFAL.py |
| Technologies Used | Python, stable-baselines3, PyTorch, NumPy, scikit-learn |
| APIs Used | POST /generate-attack, POST /apply-attack, POST /simulate_attack, FGSM/BIM logic, DQN policy inference |
| Inputs | Clean sample, epsilon value, feature constraints, UAP artifact, model output. |
| Outputs | Attacked sample, changed label or confidence, perturbation deltas, attack response JSON. |

## 3) Before vs After Changes

| Field | Details |
| --- | --- |
| Module Name | Before vs After Changes |
| Objective | Show the effect of the attack by comparing the original and adversarial predictions. |
| Problem Statement | The report and UI must clearly show what changed after the perturbation was applied. |
| Proposed Solution | Compute clean and attacked predictions side by side, highlight changed features, and present before/after metrics and labels. |
| Files Included | fastapi_ids_system.py, real_world_ids_server.py, web_dashboard_server.py, dashboard_app.py, PLANTUML_DIAGRAMS.md |
| Technologies Used | Python, PyTorch, NumPy, pandas, Plotly, Streamlit |
| APIs Used | Clean-vs-attacked comparison helpers, before/after response structures, dashboard chart rendering |
| Inputs | Clean input sample, attacked sample, model predictions, confidence values. |
| Outputs | Comparison tables, label flip indicators, feature-difference views, UI panels. |

## 4) Core Metrics

| Field | Details |
| --- | --- |
| Module Name | Core Metrics |
| Objective | Measure IDS performance and adversarial impact using standard quantitative metrics. |
| Problem Statement | The system needs objective evaluation numbers instead of only qualitative attack demonstrations. |
| Proposed Solution | Compute Accuracy, FNR, Fooling Rate, confusion matrices, and epsilon-based curves before and after attack. |
| Files Included | forward_project.py, report_significance.py, fastapi_ids_system.py, web_dashboard_server.py, dashboard_app.py |
| Technologies Used | Python, NumPy, scikit-learn, JSON, Plotly |
| APIs Used | sklearn.metrics confusion_matrix, metric aggregation routines, metrics endpoint, significance aggregation logic |
| Inputs | True labels, predicted labels, attacked predictions, multiple seed runs, run logs. |
| Outputs | Accuracy, FNR, Fooling Rate, confusion matrices, curves, summary JSON, significance reports. |

## 5) Advanced Attack Lab (Optional)

| Field | Details |
| --- | --- |
| Module Name | Advanced Attack Lab (Optional) |
| Objective | Support the research-oriented attack generation pipeline used in the original paper workflow. |
| Problem Statement | The project needs an optional experimental environment for legacy attack methods and comparison against the newer forward pipeline. |
| Proposed Solution | Preprocess data, train the older IDS baseline, pretrain the VAE, and run Gen-AAL or DFAL attack scripts for comparative analysis. |
| Files Included | data_preprocessing.py, train_ids_gbt.py, pretrain_vae.py, attack_Gen-AAL.py, attack_DFAL.py, models.py, utils.py |
| Technologies Used | Python, PyTorch, scikit-learn, NumPy, pickle, VAE, gradient boosting |
| APIs Used | Model training utilities, dataset helpers, PyTorch network layers, pickle serialization |
| Inputs | Raw CICIDS data, NBx.npy, NBy.npy, pretrained artifacts, model checkpoints. |
| Outputs | Preprocessed arrays, baseline IDS model, VAE weights, adversarial attack samples, comparison results. |

## 6) Live Drift Monitor

| Field | Details |
| --- | --- |
| Module Name | Live Drift Monitor |
| Objective | Detect distribution drift between the training baseline and live traffic samples. |
| Problem Statement | An operational IDS must warn when real-world traffic deviates from the reference distribution. |
| Proposed Solution | Track a live sample window, compare it against the training baseline with PSI and z-score style measures, and expose drift summaries in the service UI. |
| Files Included | real_world_ids_server.py, fastapi_ids_system.py, web_dashboard_server.py, dashboard_app.py |
| Technologies Used | Python, NumPy, pandas, PyTorch, FastAPI, Flask |
| APIs Used | GET /drift, live window tracking, PSI calculation, drift summary helpers |
| Inputs | Recent live samples, reference baseline arrays, feature distributions. |
| Outputs | Drift scores, top drifted features, status flags such as unavailable, insufficient_live_data, and ok. |

## 7) Benchmark Run History (Optional)

| Field | Details |
| --- | --- |
| Module Name | Benchmark Run History (Optional) |
| Objective | Store and analyze repeated experimental runs for robustness reporting. |
| Problem Statement | A project report needs historical results so the system can show consistency across seeds and benchmark settings. |
| Proposed Solution | Save timestamped run metrics, aggregate multiple runs, and generate significance summaries with mean, standard deviation, and confidence intervals. |
| Files Included | forward_project.py, report_significance.py, trained_models/forward_runs/, trained_models/forward_metrics.json, trained_models/forward_significance.json |
| Technologies Used | Python, JSON, NumPy, statistics-oriented evaluation, file-based run logging |
| APIs Used | Run-metric saving logic, report aggregation script, JSON persistence |
| Inputs | Multiple benchmark runs, seed values, evaluation metrics, run output files. |
| Outputs | Benchmark history files, aggregated summary JSON, significance report JSON. |

## Summary

These seven modules reflect the actual behavior of the repository: defense input analysis, attack simulation, before/after comparison, metric computation, optional legacy attack lab, live drift monitoring, and optional benchmark history. This structure is appropriate for a mini/major project report because it groups related files into major system capabilities instead of treating each file as a separate module.
