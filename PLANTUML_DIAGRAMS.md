# PlantUML Diagrams

## 1. Use Case Diagram

```plantuml
@startuml
left to right direction
actor "Researcher" as Researcher
actor "Security Analyst" as Analyst
actor "API Client" as APIClient

rectangle "Adversarial IDS System" {
  usecase "Run Data Preprocessing" as UC1
  usecase "Train DQN IDS Model" as UC2
  usecase "Generate UAP (FGSM/BIM)" as UC3
  usecase "Evaluate Metrics\n(Accuracy/FNR/Fooling Rate)" as UC4
  usecase "Launch Dashboard" as UC5
  usecase "Get Prediction" as UC6
  usecase "Simulate Attack" as UC7
  usecase "Upload CSV for Batch Prediction" as UC8
  usecase "View Alerts" as UC9
}

Researcher --> UC1
Researcher --> UC2
Researcher --> UC3
Researcher --> UC4
Researcher --> UC5

Analyst --> UC5
Analyst --> UC6
Analyst --> UC7
Analyst --> UC9

APIClient --> UC6
APIClient --> UC7
APIClient --> UC8
APIClient --> UC9
@enduml
```

## 2. Class Diagram

```plantuml
@startuml
class DataPreprocessor {
  - data_file_path: str
  + load_csv(): DataFrame
  + clean_invalid_rows(df): DataFrame
  + normalize(df): ndarray
  + save_arrays(x, y): void
}

class IDSEnv {
  - x: ndarray
  - y: ndarray
  - n_features: int
  + reset(seed=None): tuple
  + step(action): tuple
}

class DQNTorchAdapter {
  - model: DQN
  - device: torch.device
  + predict_label(x): int
  + q_values(x_tensor): tensor
}

class FeatureConstraint {
  - n_features: int
  - mf_idx: List<int>
  - rf_idx: List<int>
  - uf_idx: List<int>
  - recalc_rules: List<Tuple<int,int,int>>
  + apply(x, delta): tensor
}

class ForwardPipeline {
  - epsilon: float
  - uap_epochs: int
  - attack: str
  + load_preprocessed_data(): tuple
  + train_dqn_ids(): DQN
  + generate_uap(): ndarray
  + evaluate_attack(): dict
  + save_artifacts(): void
}

class IDSService {
  - model
  - n_features: int
  - alert_threshold: float
  - uap: ndarray
  - alerts: List<dict>
  + predict_one(features): dict
  + predict_batch(features_2d): dict
  + simulate_attack(features, epsilon): dict
  + sample_for_simulation(index=None): dict
}

class WebDashboardServer {
  + api_init(): json
  + api_predict(): json
  + api_apply_attack(): json
}

class RealWorldIDSServer {
  + health(): json
  + schema(): json
  + predict(): json
  + predict_batch(): json
  + predict_csv(): json
  + simulate_attack(): json
  + alerts(): json
}

ForwardPipeline --> IDSEnv : uses
ForwardPipeline --> DQNTorchAdapter : uses
ForwardPipeline --> FeatureConstraint : uses
WebDashboardServer --> ForwardPipeline : uses artifacts
RealWorldIDSServer --> IDSService : uses
IDSService --> DQNTorchAdapter : uses model policy
IDSService --> FeatureConstraint : applies UAP constraints
DataPreprocessor --> ForwardPipeline : provides NBx/NBy
@enduml
```

## 3. Activity Diagram

```plantuml
@startuml
start
:Load CSV dataset;
:Remove NaN and Inf rows;
:Normalize features to [0,1];
:Save NBx.npy and NBy.npy;

:Load preprocessed arrays;
:Train/Test split;
:Train DQN IDS model;

:Initialize UAP;
repeat
  :Select attack-class sample;
  :Apply current constrained UAP;
  if (Prediction changed?) then (yes)
    :Keep current UAP;
  else (no)
    if (Attack mode == BIM?) then (yes)
      :Generate BIM perturbation;
    else (no)
      :Generate FGSM perturbation;
    endif
    :Update and clip UAP;
    :Refine UAP with PCC objective;
  endif
repeat while (More epochs?) is (yes)

:Apply UAP on test set;
:Compute Accuracy/FNR/Fooling Rate;
:Save model, UAP, metrics;

if (Run service/dashboard?) then (yes)
  :Start Flask/Streamlit app;
  :Serve prediction and simulation APIs;
else (no)
  :End pipeline run;
endif
stop
@enduml
```

## 4. Sequence Diagram

```plantuml
@startuml
actor Analyst
participant "Web UI" as WebUI
participant "Flask Dashboard API" as DashboardAPI
participant "IDS Service" as Service
participant "DQN Model" as Model
participant "UAP Artifact" as UAP

Analyst -> WebUI : Open dashboard
WebUI -> DashboardAPI : GET /api/init
DashboardAPI --> WebUI : dataset stats, sample rows

Analyst -> WebUI : Request sample prediction
WebUI -> DashboardAPI : POST /api/predict(sample_index)
DashboardAPI -> Model : predict(sample)
Model --> DashboardAPI : label, confidence
DashboardAPI --> WebUI : prediction response

Analyst -> WebUI : Apply attack (epsilon)
WebUI -> DashboardAPI : POST /api/apply_attack
DashboardAPI -> UAP : load/scaled perturbation
DashboardAPI -> Service : simulate clean vs attacked
Service -> Model : predict(clean)
Model --> Service : clean result
Service -> Model : predict(attacked)
Model --> Service : attacked result
Service --> DashboardAPI : decision_changed, deltas, metrics
DashboardAPI --> WebUI : before/after + curves + confusion matrices
@enduml
```

## 5. Data Flow Diagram (Level 0)

```plantuml
@startuml
left to right direction
actor "Researcher/Analyst" as User
actor "API Client" as Client

rectangle "Adversarial IDS System" as System
database "Data & Artifacts\n(CSV, NBx/NBy, Model, UAP, Metrics, Alerts)" as Store

User --> System : Configure run, request analysis
Client --> System : Predict / simulate / upload CSV
System --> User : Metrics, dashboards, alerts
System --> Client : Prediction responses
System <--> Store : Read/write datasets and artifacts
@enduml
```

## 6. Data Flow Diagram (Level 1)

```plantuml
@startuml
left to right direction
actor "Researcher/Analyst" as User
actor "API Client" as Client

rectangle "P1 Data Preprocessing" as P1
rectangle "P2 IDS Training (DQN)" as P2
rectangle "P3 UAP Generation & Constraint Application" as P3
rectangle "P4 Evaluation & Metrics" as P4
rectangle "P5 API & Dashboard Services" as P5

database "D1 Raw CSV" as D1
database "D2 NBx/NBy Arrays" as D2
database "D3 Trained Model" as D3
database "D4 UAP + Metrics" as D4
database "D5 Alert Log (JSONL)" as D5

User --> P1 : Start preprocessing
P1 --> D2 : Save normalized arrays
D1 --> P1 : Load source data

User --> P2 : Start training
D2 --> P2 : Train/test data
P2 --> D3 : Save DQN model

User --> P3 : Start attack generation
D2 --> P3 : Training samples
D3 --> P3 : IDS model policy
P3 --> D4 : Save UAP

User --> P4 : Run evaluation
D2 --> P4 : Test data
D3 --> P4 : Model
D4 --> P4 : UAP
P4 --> D4 : Save metrics

Client --> P5 : Predict / batch / CSV / simulate
User --> P5 : Dashboard interactions
D2 --> P5 : Simulation samples
D3 --> P5 : Inference model
D4 --> P5 : UAP + metrics
P5 --> D5 : Append alerts
P5 --> Client : API responses
P5 --> User : UI data
@enduml
```

## 7. Flowchart

```plantuml
@startuml
start
:Run data_preprocessing.py;
if (NBx.npy and NBy.npy created?) then (yes)
  :Run forward_project.py;
  :Train DQN model;
  :Generate UAP;
  :Evaluate clean vs attacked outputs;
  :Save dqn_ids_model.zip, uap.npy, forward_metrics.json;
  if (Need visualization?) then (yes)
    :Run web_dashboard_server.py or dashboard_app.py;
    :Inspect metrics and sample behavior;
  endif
  if (Need API service?) then (yes)
    :Run real_world_ids_server.py;
    :Use /predict, /simulate_attack, /alerts;
  endif
  stop
else (no)
  :Fix dataset path/format;
  stop
endif
@enduml
```

## 8. Proposed System Architecture Diagram

```plantuml
@startuml
left to right direction

node "Frontend Layer" {
  component "Web Dashboard UI\n(web/index.html + app.js)" as WebUI
  component "Service UI\n(service_ui/index.html + app.js)" as ServiceUI
  component "Streamlit Dashboard\n(dashboard_app.py)" as StreamlitUI
}

node "Backend/API Layer" {
  component "Web Dashboard Server\n(Flask)" as WebServer
  component "Real-World IDS Server\n(Flask API)" as APIServer
}

node "ML Core Layer" {
  component "Forward Pipeline\n(forward_project.py)" as Pipeline
  component "DQN IDS Model\n(stable-baselines3)" as DQN
  component "UAP Generator\n(FGSM/BIM + PCC + Constraints)" as UAPGen
  component "Preprocessing\n(data_preprocessing.py)" as Prep
}

database "Artifact Storage\nNBx/NBy, dqn_ids_model.zip,\nuap.npy, forward_metrics.json,\nids_alerts.jsonl" as Storage

WebUI --> WebServer : HTTP JSON
ServiceUI --> APIServer : HTTP JSON
StreamlitUI --> Storage : Read artifacts

WebServer --> DQN : Inference
WebServer --> Storage : Read metrics/artifacts

APIServer --> DQN : Inference
APIServer --> Storage : Read UAP/simulation data
APIServer --> Storage : Write alert logs

Prep --> Storage : Save NBx/NBy
Pipeline --> DQN : Train/save model
Pipeline --> UAPGen : Generate perturbation
UAPGen --> Storage : Save uap.npy
Pipeline --> Storage : Save metrics
Pipeline --> Storage : Read NBx/NBy
@enduml
```

## 9. Gantt Chart

```plantuml
@startgantt
Project starts 2026-01-01
printscale weekly
saturday are closed
sunday are closed

[Requirement Gathering] as [RG] lasts 2 weeks
[Design] as [DS] lasts 2 weeks
[Development] as [DV] lasts 6 weeks
[Testing] as [TS] lasts 3 weeks
[Deployment] as [DP] lasts 1 weeks

[DS] starts after [RG]
[DV] starts after [DS]
[TS] starts after [DV]
[DP] starts after [TS]
@endgantt
```
