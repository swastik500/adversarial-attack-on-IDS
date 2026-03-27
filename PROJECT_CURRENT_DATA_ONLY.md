# Adversarial Attack on IDS - Technical Report (Current State Only)

This document contains only project-related technical content that is verifiable from the current repository state. It excludes personal details, diagrams, acknowledgements, approval/certificate pages, and figure/table lists.

# 1. Abstract
This project investigates adversarial machine learning against Intrusion Detection Systems (IDS) using network-flow data derived from the CICIDS2017 source files present in the repository. The core objective is to examine how model decisions change when carefully controlled perturbations are applied to normalized tabular feature vectors representing network traffic behavior.

The repository currently contains two practical tracks:

- A legacy active-learning and adversarial-generation track, including Gen-AAL and DFAL style scripts.
- A forward pipeline that trains a Deep Q-Network (DQN) IDS policy, generates a Universal Adversarial Perturbation (UAP), applies feature constraints, refines perturbations using a correlation-based objective, and evaluates prediction shifts under attack.

At the implementation level, the project emphasizes the following practical capabilities:

- End-to-end preprocessing from CSV to model-ready arrays.
- Train/test workflow for IDS decision modeling.
- Attack generation through gradient-based perturbation mechanisms (FGSM/BIM in the forward pipeline).
- Constraint-aware perturbation application to keep generated examples bounded and structurally guided.
- Evaluation through security-relevant metrics such as false negative rate (FNR), which is critical for IDS because missed attacks are more dangerous than generic classification errors.
- Deployment-oriented APIs and dashboards to inspect behavior interactively, simulate attacks, and observe decision transitions before and after perturbation.

The system should be interpreted as a research and prototyping codebase rather than a finished enterprise IDS platform. Some modules are mature and runnable, while other elements are legacy or partially integrated. This report therefore presents only what is currently available and verifiable.

# 2. Abbreviations
- CF : Collaborative Filtering
- CBF : Content Based Filtering
- API : Application Programming Interface
- SQL : Structured Query Language
- URL : Uniform Resource Locator
- HTTP : HyperText Transfer Protocol

Additional project-relevant abbreviations detected from the codebase and README files:

- IDS : Intrusion Detection System
- ML : Machine Learning
- DRL : Deep Reinforcement Learning
- DQN : Deep Q-Network
- UAP : Universal Adversarial Perturbation
- FGSM : Fast Gradient Sign Method
- BIM : Basic Iterative Method
- PCC : Pearson Correlation Coefficient
- FNR : False Negative Rate
- GBT : Gradient Boosting Tree
- VAE : Variational Autoencoder
- CSV : Comma-Separated Values
- JSONL : JSON Lines

# 3. Introduction
## Background
Intrusion Detection Systems analyze network behavior and classify traffic as benign or malicious. In modern pipelines, this often becomes a supervised or reinforcement-learning classification task over tabular flow features extracted from packet traces. Once ML is introduced, IDS gains adaptability and pattern-learning capacity but also inherits adversarial vulnerability.

Adversarial vulnerability in this context means that small, often norm-bounded perturbations in feature space can push the model toward incorrect decisions. For an IDS, one of the highest-risk failure modes is the malicious-to-benign flip, because it can allow hostile traffic to bypass detection.

This project is positioned around that exact risk: not only measuring baseline IDS behavior, but deliberately constructing perturbations and quantifying how much model behavior changes.

## Relevance
The project matters for both research and practical security engineering.

- Security relevance:
  - IDS systems are frequently evaluated in terms of accuracy, but operational impact depends heavily on failure type.
  - A moderate fooling rate or a high post-attack FNR can indicate severe operational risk even when aggregate metrics appear acceptable.
- ML robustness relevance:
  - Adversarial ML on tabular network data is less standardized than image-based adversarial benchmarks.
  - Constraint handling (modifiable, recalculated, unmodified features) is important for realism in structured tabular domains.
- Engineering relevance:
  - The repository includes service endpoints and dashboards, enabling practical observation rather than only offline notebook-style analysis.

## Problem Overview
The technical problem solved by the current code can be summarized in stages:

1. Convert raw tabular flow data to stable, normalized numeric arrays.
2. Build an IDS decision model with reproducible train/test split and evaluation.
3. Generate perturbations that are bounded, transferable, and reusable (UAP).
4. Apply feature constraints so perturbations are not completely unconstrained vector noise.
5. Compare model behavior before and after attack using security-focused metrics.
6. Expose outputs through API and dashboard layers for operational-style inspection.

The repository does this in a practical, script-driven way. It does not yet present a single monolithic production framework; instead, it provides multiple interoperable modules at different maturity levels.

# 4. Literature Review
## Related Work
Repository evidence indicates two relevant lines of related work:

- The top-level code references the paper titled Generative Adversarial Attacks Against Intrusion Detection Systems Using Active Learning.
- The nested adversarial_ml_ids subproject references published work on model evasion attacks on IDS.

From a technical standpoint, the implemented scripts align with common adversarial-ML patterns:

- Active query and retraining loops in legacy scripts.
- Gradient-guided perturbation generation.
- Evaluation of attack success and perturbation behavior.
- Use of tabular flow datasets rather than image benchmarks.

Detailed bibliographic expansion beyond these repository references is not available in the currently accessible project files.

Not implemented yet

## Existing System
The current repository effectively contains two existing systems in parallel:

1. Legacy research stack
   - Includes DFAL/Gen-AAL style scripts, GBT and VAE components, utility modules, and data handling paths.
   - Strength: rich research experimentation logic.
   - Limitation: integration and packaging are script-centric; end-to-end deployment contracts are limited.

2. Forward pipeline and service stack
   - Includes preprocessing, DQN training, UAP generation, metric export, dashboard APIs, and real-world style service endpoints.
   - Strength: clearer operational flow and deployment-style access via Flask endpoints and UI.
   - Limitation: still research-grade in validation, hardening, and lifecycle tooling.

Key limitations visible from current implementation:

- Dataset coverage in default preprocessing points to a specific CICIDS CSV file path rather than a broad dataset ingestion framework.
- Model evaluation reporting is available but not yet accompanied by extensive experiment tracking metadata.
- Robustness analyses are present but not yet unified into a formal benchmark suite with repeatable multi-run statistics.
- Alerting is in-memory plus JSONL logging; durable queueing, retention policy, and SIEM integration are not present.

## Problem Statement
The project addresses this concrete problem:

How can an IDS model trained on normalized network-flow features be stress-tested against realistic bounded perturbations, and how can resulting behavior shifts be measured through attack-relevant metrics and exposed through practical interfaces?

The implemented answer is a multi-module pipeline that:

- Trains IDS decision logic.
- Generates and applies adversarial perturbations under constraints.
- Computes pre/post behavior metrics.
- Provides API-level and dashboard-level visibility into outcomes.

# 5. Requirements
## Software Requirements
The following requirements are directly supported by repository files and imports.

Core language/runtime:

- Python environment (the project scripts are Python-based).

Primary packages from requirements_forward.txt and code imports:

- numpy
  - Purpose: numerical arrays, vectorized perturbation operations, metrics support.
- pandas
  - Purpose: CSV ingestion, typed tabular transformations, upload parsing in API.
- scikit-learn
  - Purpose: train/test split, metrics utilities, confusion matrix, and legacy GBT model path.
- torch
  - Purpose: tensor operations, gradient computation, adversarial update paths, model-related operations.
- torchvision
  - Purpose: listed dependency used by legacy scripts.
- gymnasium
  - Purpose: custom RL environment for IDS decision process.
- stable-baselines3
  - Purpose: DQN training and inference engine.
- flask
  - Purpose: API and web serving for dashboard and service interfaces.

Project scripts and services:

- data_preprocessing.py
  - Normalizes CSV and writes canonical arrays.
- forward_project.py
  - Runs DRL IDS plus UAP pipeline and saves artifacts.
- dashboard_app.py
  - Streamlit visualization flow.
- web_dashboard_server.py
  - Flask API for browser-based dashboard using web frontend.
- real_world_ids_server.py
  - Flask service for prediction, attack simulation, and alert handling.

Frontend assets:

- web and service_ui folders containing HTML/CSS/JS clients.

Operational file dependencies:

- Input arrays: NBx.npy, NBy.npy.
- Artifacts: trained_models/dqn_ids_model.zip, trained_models/uap.npy, trained_models/forward_metrics.json.

## Hardware Requirements
The code supports both CPU and GPU execution paths.

Minimum practical setup for functional execution:

- Multi-core CPU.
- Sufficient RAM for loading normalized tabular arrays and batched inference.
- Disk space for dataset CSV, model artifacts, and logs.

Recommended setup for faster experimentation:

- CUDA-capable GPU for PyTorch operations in training/attack loops.
- Higher RAM for large evaluation slices and faster dataframe operations.

Reasoning based on implementation:

- Several scripts branch on CUDA availability.
- DQN training timesteps and iterative perturbation generation are computationally heavier than pure inference.
- Dashboard and API services are lightweight compared with model training.

Exact benchmark hardware profile in the accessible project files:

Not implemented yet

# 6. Methodology
The current working methodology is best described as a layered pipeline with reusable artifacts between phases.

Phase A: Data preparation

1. Load the source CICIDS CSV selected in preprocessing script.
2. Remove invalid rows containing NaN values.
3. Remove rows containing infinite values.
4. Perform MinMax scaling so each feature lies within [0, 1].
5. Generate binary labels and write NumPy arrays.
6. Save canonical and compatibility file names.

Phase B: IDS model setup

1. Read NBx.npy and NBy.npy arrays.
2. Convert label convention for attack-positive evaluation in forward pipeline.
3. Apply stratified train/test split for balanced class representation.
4. Optionally create a balanced subset for DQN training to limit class skew effects.
5. Train DQN policy using custom Gymnasium environment with discrete action space.

Phase C: Adversarial perturbation generation

1. Initialize UAP vector as zeros.
2. Iterate over attack-class samples.
3. Apply existing UAP to each sample using feature constraints.
4. Check if prediction changes.
5. If unchanged, produce local perturbation using FGSM or BIM.
6. Aggregate local perturbation into global UAP with epsilon clipping.
7. Refine UAP using objective combining misclassification pressure and PCC term.
8. Repeat for configured epochs.

Phase D: Constraint-aware application

1. Define feature groups:
   - MF for modifiable features.
   - RF for recomputed features.
   - UF for fixed/unmodified features.
2. Apply perturbation only where allowed.
3. Recompute dependent features via ratio rules.
4. Clip output back to [0, 1].

Phase E: Evaluation and persistence

1. Predict on clean test set.
2. Predict on attacked test set.
3. Compute accuracy before/after.
4. Compute FNR before/after.
5. Compute fooling rate.
6. Persist model, UAP, and metrics to trained_models folder.

Phase F: Interaction and service layer

1. Expose APIs for single prediction, batch prediction, CSV prediction, sample fetch, and attack simulation.
2. Render dashboards that display dataset metadata, predictions, perturbation impact, confusion matrices, and metric curves.
3. Log alerts to JSONL for high-confidence attack predictions.

Methodological observations about maturity:

- Core train-attack-evaluate flow is implemented and executable.
- Production hardening topics such as auth, role-based control, encrypted artifact lifecycle, and distributed serving are outside current scope.

# 7. Algorithm
This section presents algorithmic logic derived from implemented code paths.

Algorithm A: Preprocessing

Input:
- Raw CSV flow file

Output:
- NBx.npy (normalized features)
- NBy.npy (binary labels)

Steps:

1. Read CSV rows into tabular structure.
2. Split feature matrix and label column.
3. Filter out rows with NaN in feature columns.
4. Filter out rows with Inf in feature columns.
5. Apply MinMax scaling to all feature columns.
6. Encode BENIGN to binary label convention used by script.
7. Save output arrays to disk.

Algorithm B: DQN IDS training

Input:
- x_train, y_train

Output:
- Trained DQN model

Steps:

1. Build custom environment where observation is feature vector and action is class prediction.
2. Use reward +1 for correct action and -1 for incorrect action.
3. Configure DQN with MLP policy and replay/hyperparameter settings.
4. Train for specified timesteps.
5. Return trained policy.

Algorithm C: UAP generation with optional FGSM/BIM update

Input:
- Trained model adapter
- Training samples
- Constraints
- epsilon

Output:
- Universal perturbation vector

Steps:

1. Initialize universal perturbation delta = 0.
2. Iterate through attack-class samples for multiple epochs.
3. Apply constrained delta to sample.
4. If prediction unchanged:
   - Compute sample-level perturbation via FGSM or BIM.
   - Update universal delta with sample perturbation.
   - Clip delta into [-epsilon, +epsilon].
5. Refine delta with PCC-oriented optimization step:
   - Build differentiable adversarial sample path.
   - Compute cross-entropy term relative to clean prediction.
   - Compute Pearson correlation between selected Q-value views.
   - Optimize objective combining both terms.
6. Repeat until epoch completion.
7. Return final delta.

Algorithm D: Constraint application

Input:
- Sample vector x
- Perturbation delta
- Group definitions and recalc rules

Output:
- Constrained adversarial sample x_adv

Steps:

1. Set x_adv as copy of x.
2. For MF indices, apply additive perturbation from delta.
3. For UF indices, keep original values.
4. For each RF target, recompute from corresponding numerator/denominator rule.
5. Clamp final vector to [0, 1].

Algorithm E: Attack evaluation

Input:
- y_true
- Predictions before and after attack

Output:
- accuracy_before
- accuracy_after
- fnr_before
- fnr_after
- fooling_rate

Steps:

1. Compute standard accuracy on clean and attacked predictions.
2. Compute FNR as FN / (TP + FN) with denominator guard.
3. Compute fooling rate as fraction of samples where predicted label changed.
4. Return metric dictionary and persist to JSON file.

# 8. Project Plan
The current repository reflects an incremental plan executed in phases rather than a formal chart artifact.

Phase 1: Dataset and baseline preparation

- Add preprocessing for CICIDS-derived CSV.
- Build canonical NumPy arrays and compatibility outputs.
- Establish reusable data representation consumed by multiple scripts.

Phase 2: Legacy adversarial experimentation

- Implement/retain DFAL and Gen-AAL script workflows.
- Support GBT and VAE-related experimentation loops.
- Explore active-learning style query and retraining behavior.

Phase 3: Forward DRL-focused pipeline

- Introduce DQN IDS environment and training routine.
- Integrate FGSM/BIM-driven UAP construction.
- Add constraint-based perturbation handling and PCC refinement.
- Persist key artifacts for downstream services.

Phase 4: Evaluation and visualization

- Add metric persistence and confusion-matrix-ready outputs.
- Build Streamlit dashboard path for rapid analysis.
- Build Flask plus frontend dashboard for browser interaction.

Phase 5: Service-style interface

- Build inference service with health/schema/predict endpoints.
- Add batch and CSV prediction paths.
- Add simulation sample and attack simulation endpoints.
- Add alert generation and JSONL logging.

Phase 6: Integration and hardening status

- Current state indicates functional modules, but complete unification of all research and service tracks into one orchestrated production workflow is still pending.

Detailed sprint log, milestone dates, effort estimation, and risk matrix:

Not implemented yet

# 9. Project Analysis (TEXT ONLY)
## Use Case Description
The current build supports multiple user interaction modes.

Use case 1: Researcher executes full adversarial pipeline

- Actor: ML security researcher.
- Goal: Train IDS model, generate UAP, quantify attack impact.
- Flow:
  1. Run preprocessing.
  2. Run forward pipeline with selected attack mode and hyperparameters.
  3. Inspect saved metrics and artifacts.
  4. Validate behavior in dashboards and API.
- System response:
  - Produces model artifact, UAP, and metrics.
  - Enables visualization and simulation with generated outputs.

Use case 2: Analyst performs interactive dashboard analysis

- Actor: security analyst or evaluator.
- Goal: Understand prediction behavior before/after perturbation.
- Flow:
  1. Launch dashboard server.
  2. Load dataset metadata and sample rows.
  3. Choose sample index and epsilon.
  4. Trigger attack simulation and review changed features.
  5. Compare metrics and confusion matrices.
- System response:
  - Returns detailed before/after predictions, confidence, and impact visual outputs.

Use case 3: Operator uses real-world style IDS API

- Actor: API consumer or integration engineer.
- Goal: Run clean and attacked inference workflows through HTTP.
- Flow:
  1. Start real_world_ids_server.
  2. Query health/schema endpoints.
  3. Submit feature payload for prediction.
  4. Optionally run simulate_attack endpoint.
  5. Inspect alerts and upload CSV for batch scoring.
- System response:
  - Provides predictions, confidence scores, attack simulation responses, and alert logs.

Use case 4: Batch scoring and upload workflow

- Actor: evaluator with pre-collected feature rows.
- Goal: score multiple rows quickly.
- Flow:
  1. Build batch JSON or upload CSV.
  2. Call batch endpoint.
  3. Receive per-row and aggregate attack ratio output.

## Functional Analysis
The functional behavior that is currently implemented can be decomposed as follows.

Data functions:

- File loading from CSV and NumPy formats.
- Data cleaning and normalization.
- Label conversion logic for attack-positive evaluation mode.

Training functions:

- Custom RL environment exposing observation and action spaces.
- DQN training loop and model serialization.

Attack functions:

- Local perturbation generation using FGSM/BIM.
- Universal perturbation accumulation.
- Constraint-based application and recomputation of dependent feature groups.
- PCC-influenced refinement pass.

Evaluation functions:

- Clean and adversarial prediction generation.
- Metric computation with security orientation (not only accuracy).
- Confusion matrix generation in visualization layers.

Service functions:

- Health and schema discoverability.
- Single, batch, and CSV prediction APIs.
- Attack simulation APIs.
- Alert creation and persistent append logging.

UI functions:

- Dataset and feature row display.
- Prediction and confidence reporting.
- Attack parameter control and outcome display.
- Curves and feature-impact visualization.

Non-functional status visible from code:

- Input clipping to [0, 1] implemented as basic safety in service logic.
- Error handling returns JSON error payloads on request failures.
- No explicit authentication/authorization in current API services.
- No distributed scaling, queueing, or observability stack integration.

# 10. System Design (TEXT ONLY)
## Architecture Explanation
The current system is modular and script-driven. The architecture can be described in six logical layers.

Layer 1: Data layer

- Stores source CSV and generated NumPy arrays.
- Provides normalized feature vectors and binary labels.

Layer 2: Model and attack core

- Contains training and attack algorithms.
- Includes DQN policy logic, UAP generation, and constraint handling.

Layer 3: Artifact layer

- Persists model zip file, perturbation vector, and metric JSON.
- Acts as the handoff mechanism to dashboards and services.

Layer 4: Service layer

- Exposes prediction and simulation APIs through Flask.
- Encapsulates request coercion, validation, model inference, and response formatting.

Layer 5: Presentation layer

- Includes Streamlit and static web UIs.
- Converts API/model outputs into analyst-friendly views.

Layer 6: Logging and operational support

- Maintains in-memory alert list and JSONL log append path.

## Data Flow Explanation in Words
End-to-end data flow for training and evaluation:

1. Raw CSV is read by preprocessing script.
2. Cleaned and normalized arrays are written to NumPy files.
3. Forward pipeline loads arrays and splits data.
4. DQN model is trained using custom IDS environment.
5. UAP vector is generated and refined.
6. UAP is applied to test samples under constraints.
7. Metrics are computed and saved.
8. Model and UAP artifacts are persisted.

Data flow for service prediction:

1. Client sends feature payload to API.
2. API coerces and validates payload shape.
3. Input is clipped and passed to model.
4. Prediction and confidence are computed from Q-value softmax.
5. Alert policy evaluates threshold for attack predictions.
6. Response JSON is returned; alerts may be logged.

Data flow for attack simulation endpoint:

1. Client submits clean features and epsilon.
2. Service scales stored UAP to requested epsilon.
3. Service creates adversarial version through clipping.
4. Service predicts clean and adversarial outputs.
5. Service computes top feature deltas and returns comparison payload.

Data flow for dashboard interaction:

1. Browser loads initial metadata via init/health/schema endpoints.
2. User triggers sample load and attack actions.
3. Backend returns metrics, matrices, and feature-level changes.
4. Frontend renders tables, cards, and curves.

# 11. Implementation (MOST IMPORTANT SECTION)
This section describes only what is presently built and observable in the repository.

## Completed Modules
1. Data preprocessing module

- Implemented in data_preprocessing.py.
- Reads CICIDS CSV path configured in script.
- Performs invalid-row filtering (NaN and Inf removal).
- Applies MinMax normalization.
- Saves four NumPy outputs for current and compatibility usage.

2. Forward DRL plus UAP pipeline

- Implemented in forward_project.py.
- Includes:
  - DQN training with Gymnasium environment.
  - Balanced subset creation for training.
  - FGSM and BIM local update options.
  - UAP generation loop across attack samples.
  - FeatureConstraint logic with MF/RF/UF handling.
  - PCC-guided optimization pass.
  - Evaluation and persistence of metrics/artifacts.

3. Artifact persistence and reuse

- trained_models directory currently includes model, UAP, and metrics outputs.
- These artifacts are consumed by service and dashboard modules.

4. Web dashboard backend and frontend

- Backend in web_dashboard_server.py.
- Frontend in web folder.
- Supports:
  - Initialization payload (dataset stats and sample rows).
  - Per-sample prediction endpoint.
  - Attack simulation endpoint returning feature deltas, metrics, confusion matrices, and curves.

5. Streamlit dashboard

- Implemented in dashboard_app.py.
- Supports dataset panel, prediction panel, attack controls, metric cards, curve generation, and confusion matrices.

6. Real-world IDS service API

- Implemented in real_world_ids_server.py with frontend in service_ui.
- Supports health/schema discovery, single and batch prediction, CSV upload scoring, simulation sample retrieval, attack simulation, and alerts.
- Includes configurable host/port/model/uap/log paths and alert threshold.

7. Alert generation and logging

- High-confidence attack predictions trigger alert object generation.
- Alerts are stored in memory with bounded list size and appended to JSONL log file.

8. Legacy adversarial research modules retained

- attack_Gen-AAL.py and attack_DFAL.py remain available.
- Legacy dependencies and helper modules are still part of repository state.

## Modules In Progress (with current state)
1. Full integration between legacy and forward tracks

- Current state:
  - Both tracks exist.
  - No single orchestrator unifies all training, attack, evaluation, and serving modes into one standardized command interface.

2. Robust experiment management

- Current state:
  - Metrics are saved for forward runs.
  - Comprehensive run registry, seed sweep framework, and comparative experiment database are not visible.

3. Production-grade API hardening

- Current state:
  - Core functional endpoints are implemented.
  - Security controls such as authentication, authorization, and traffic-rate governance are not implemented in current code.

4. Dataset generalization framework

- Current state:
  - Preprocessing script is wired to a specific dataset path.
  - Generalized multi-source ingestion and schema adaptation workflow is not fully implemented.

5. Maturity of reported metrics interpretation

- Current state:
  - Metrics are available and usable.
  - Interpretation layer for multi-run statistical confidence and significance testing is not present.

## Not Implemented Yet
1. End-to-end automated CI pipeline for training, evaluation, and deployment.
2. Built-in authentication and role control for public-facing APIs.
3. Integrated model registry with versioning policy and rollback controls.
4. Formal adversarial benchmark harness covering many attack families in one framework.
5. Real-time stream ingestion from live network collectors.
6. Native SIEM connector modules and enterprise alert pipeline adapters.
7. Formal report export generator from run artifacts.
8. Documented reproducibility matrix covering hardware variance and multi-seed confidence intervals.

# 12. Sample Code (if available)
Real code snippets from currently available modules are included below.

Snippet 1: Label conversion and split logic in forward pipeline

    x = np.load(x_path).astype(np.float32)
    y_raw = np.load(y_path).astype(np.int64)

    # Existing preprocessing uses BENIGN=1 and ATTACK=0. Convert to ATTACK=1 for evaluation.
    if set(np.unique(y_raw)).issubset({0, 1}):
        y_attack = 1 - y_raw
    else:
        y_attack = (y_raw != 0).astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_attack,
        test_size=test_ratio,
        random_state=seed,
        stratify=y_attack,
    )

Snippet 2: Core attack evaluation metrics

    acc_before = float(accuracy_score(y_test, y_pred_before))
    acc_after = float(accuracy_score(y_test, y_pred_after))

    fnr_before = false_negative_rate(y_test, y_pred_before)
    fnr_after = false_negative_rate(y_test, y_pred_after)

    fooling_rate = float(np.mean(y_pred_before != y_pred_after))

Snippet 3: Real-world API endpoint list response

    @app.get("/api")
    def api_info():
        return jsonify(
            {
                "service": "real-world-ids-api",
                "status": "ok",
                "time": utc_now_iso(),
                "endpoints": [
                    "/health",
                    "/schema",
                    "/predict",
                    "/predict_batch",
                    "/predict_csv",
                    "/simulation_sample",
                    "/simulate_attack",
                    "/alerts",
                ],
            }
        )

Snippet 4: Preprocessing output persistence

    np.save('NBx', X_scaled)
    np.save('NBy', Y_labels)

    # Keep legacy filenames for compatibility with previous runs.
    np.save('NBx_test', X_scaled)
    np.save('NBy_test', Y_labels)

# 13. Result Analysis
Current measurable outputs are available from trained_models/forward_metrics.json.

Observed values in repository at report generation time:

- Accuracy before attack: 0.9419375114579787
- Accuracy after attack: 0.9897336511012754
- FNR before attack: 0.020460358056265986
- FNR after attack: 1.0
- Fooling rate: 0.06790980279181835

Behavioral interpretation based strictly on these values and implemented metrics:

- The post-attack FNR being 1.0 indicates all attack-class examples in evaluation were predicted as benign after perturbation in that measured run slice. This is a severe IDS failure pattern because it corresponds to complete miss rate on malicious class under the evaluated condition.
- The increase in accuracy after attack, when seen together with FNR=1.0, suggests class-distribution and decision-shift effects where aggregate accuracy does not represent security quality. This confirms why FNR and class-sensitive analysis are necessary in IDS contexts.
- A fooling rate around 0.0679 means the predicted label changed for a subset of samples; even partial flipping can be operationally critical if concentrated in malicious traffic.

What currently works:

- End-to-end generation of artifacts and metrics from preprocessing through attack evaluation.
- Interactive simulation and before/after comparison through APIs and dashboards.
- Alerting behavior for high-confidence attack predictions in service mode.

What currently does not fully work or is not proven from available evidence:

- Stable robustness guarantees across datasets, seeds, and hardware profiles.
- Fully unified benchmark reporting for legacy and forward paths.
- Comprehensive enterprise deployment reliability features.

# 14. Conclusion
Based on current repository state, the project successfully demonstrates a practical adversarial-evaluation framework for IDS with the following concrete achievements:

- Data cleaning and normalization pipeline.
- Trainable DQN-based IDS model path.
- Constraint-aware UAP generation and application.
- Security-oriented evaluation metrics including FNR and fooling rate.
- Usable dashboards and HTTP services for simulation and analysis.

The project should currently be considered a strong research and prototyping platform, not a fully completed production security product. It already provides meaningful technical value for adversarial analysis of IDS behavior, while leaving several integration and hardening tasks for future development.

# 15. Future Scope
The following extensions are realistic next steps based on current architecture and code organization.

1. Unified orchestration

- Build one command-driven pipeline that can select legacy or forward mode, execute full runs, and emit standardized reports.

2. Strong experiment tracking

- Add multi-seed run automation.
- Store run metadata (seed, hyperparameters, dataset slice, artifact hashes).
- Support comparative robustness dashboards over many runs.

3. Broader attack library

- Add additional tabular attacks under same evaluation contract.
- Standardize attack interface to compare effectiveness and perturbation cost.

4. Data generalization

- Extend preprocessing to support multiple CICIDS splits and additional IDS datasets.
- Add schema mapping and validation layer for cross-dataset portability.

5. Production hardening of APIs

- Add authentication, authorization, request quotas, structured audit logging, and secure configuration handling.

6. Alerting ecosystem integration

- Export alerts to external pipelines (message brokers, SIEM adapters) with reliable delivery semantics.

7. Model lifecycle controls

- Introduce artifact versioning, model registry, staged deployment, and rollback workflows.

8. Statistical robustness reporting

- Add confidence intervals and significance tests for pre/post attack metrics.
- Include per-class drift and threshold calibration analysis.

9. Real-time and streaming evaluation

- Integrate with streaming feature pipelines for near real-time adversarial monitoring.

10. Documentation and reproducibility package

- Generate machine-readable run manifests and one-command environment checks for reproducible research sharing.
