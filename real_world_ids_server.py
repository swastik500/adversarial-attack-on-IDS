import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_from_directory


def get_dqn_cls():
    module = importlib.import_module("stable_baselines3")
    return module.DQN


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model(model_base_path: str):
    dqn_cls = get_dqn_cls()
    if os.path.exists(model_base_path):
        return dqn_cls.load(model_base_path)
    zip_path = f"{model_base_path}.zip"
    if os.path.exists(zip_path):
        return dqn_cls.load(zip_path)
    raise FileNotFoundError(
        f"Could not find DQN model at '{model_base_path}' or '{zip_path}'. "
        "Run forward_project.py first to generate trained model artifacts."
    )


def get_q_probs(model, features_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    device = model.policy.device
    x = torch.tensor(features_2d, dtype=torch.float32, device=device)

    with torch.no_grad():
        q_vals = model.policy.q_net(x)
        probs = torch.softmax(q_vals, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values

    return pred.cpu().numpy().astype(np.int64), conf.cpu().numpy().astype(np.float32)


def coerce_features(payload: Dict, n_features: int) -> np.ndarray:
    # Supported formats:
    # 1) {"features": [..]}
    # 2) {"features": {"f0": .., "f1": ..}}
    features = payload.get("features")
    if features is None:
        raise ValueError("Missing 'features' in request body.")

    if isinstance(features, list):
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("'features' list must be 1D.")
        if arr.shape[0] != n_features:
            raise ValueError(f"Expected {n_features} features, got {arr.shape[0]}.")
        return arr

    if isinstance(features, dict):
        row = []
        for i in range(n_features):
            key = f"f{i}"
            if key not in features:
                raise ValueError(f"Missing feature key '{key}'.")
            row.append(float(features[key]))
        return np.asarray(row, dtype=np.float32)

    raise ValueError("'features' must be either a list or object keyed by f0..fN.")


def coerce_batch(payload: Dict, n_features: int) -> np.ndarray:
    items = payload.get("items")
    if items is None or not isinstance(items, list) or len(items) == 0:
        raise ValueError("'items' must be a non-empty list.")

    rows = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each batch item must be an object with 'features'.")
        rows.append(coerce_features(item, n_features))

    out = np.stack(rows, axis=0)
    if out.ndim != 2 or out.shape[1] != n_features:
        raise ValueError("Malformed batch features.")
    return out


def coerce_csv_dataframe(df: pd.DataFrame, n_features: int) -> np.ndarray:
    # Accepted formats:
    # 1) columns f0..fN
    # 2) exactly N numeric columns in any names/order
    expected_cols = [f"f{i}" for i in range(n_features)]

    if all(col in df.columns for col in expected_cols):
        values = df[expected_cols].to_numpy(dtype=np.float32)
        return values

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == n_features:
        return numeric_df.to_numpy(dtype=np.float32)

    raise ValueError(
        f"CSV must include either columns f0..f{n_features - 1} or exactly {n_features} numeric columns."
    )


def to_label_name(v: int) -> str:
    return "Attack" if int(v) == 1 else "Benign"


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def load_optional_uap(uap_path: str, n_features: int) -> np.ndarray:
    if not os.path.exists(uap_path):
        return None

    uap = np.load(uap_path).astype(np.float32)
    if uap.ndim != 1 or uap.shape[0] != n_features:
        return None
    return uap


def load_optional_sim_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None

    x = np.load(x_path).astype(np.float32)
    y_raw = np.load(y_path).astype(np.int64)
    if x.ndim != 2:
        return None, None

    if set(np.unique(y_raw)).issubset({0, 1}):
        y_attack = 1 - y_raw
    else:
        y_attack = (y_raw != 0).astype(np.int64)

    if y_attack.ndim != 1 or y_attack.shape[0] != x.shape[0]:
        return None, None

    return x, y_attack


class IDSService:
    def __init__(self, model_path: str, alert_threshold: float, log_path: str, uap_path: str, sim_x_path: str, sim_y_path: str):
        self.model = load_model(model_path)
        self.n_features = int(self.model.observation_space.shape[0])
        self.alert_threshold = float(alert_threshold)
        self.log_path = log_path
        self.lock = Lock()
        self.alerts: List[Dict] = []

        self.uap = load_optional_uap(uap_path, self.n_features)
        self.sim_x, self.sim_y = load_optional_sim_data(sim_x_path, sim_y_path)

    def predict_one(self, features: np.ndarray) -> Dict:
        x = clamp01(features.astype(np.float32)).reshape(1, -1)
        with self.lock:
            pred, conf = get_q_probs(self.model, x)

        pred_v = int(pred[0])
        conf_v = float(conf[0])
        label = to_label_name(pred_v)

        item = {
            "timestamp": utc_now_iso(),
            "prediction": pred_v,
            "prediction_label": label,
            "confidence": conf_v,
        }

        if pred_v == 1 and conf_v >= self.alert_threshold:
            alert = {
                "timestamp": item["timestamp"],
                "severity": "high" if conf_v >= min(0.95, self.alert_threshold + 0.2) else "medium",
                "prediction": pred_v,
                "prediction_label": label,
                "confidence": conf_v,
                "message": "Potential intrusion detected by IDS model.",
            }
            self.alerts.append(alert)
            self.alerts = self.alerts[-500:]
            self.append_alert_log(alert)
            item["alert"] = alert
        else:
            item["alert"] = None

        return item

    def predict_batch(self, features_2d: np.ndarray) -> Dict:
        x = clamp01(features_2d.astype(np.float32))
        with self.lock:
            pred, conf = get_q_probs(self.model, x)

        records = []
        attack_count = 0
        for i in range(x.shape[0]):
            p = int(pred[i])
            c = float(conf[i])
            rec = {
                "index": i,
                "prediction": p,
                "prediction_label": to_label_name(p),
                "confidence": c,
            }
            if p == 1:
                attack_count += 1
            records.append(rec)

        return {
            "count": int(x.shape[0]),
            "attack_count": int(attack_count),
            "attack_ratio": float(attack_count / max(1, x.shape[0])),
            "records": records,
        }

    def append_alert_log(self, alert: Dict) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")

    def sample_for_simulation(self, index: int = None) -> Dict:
        if self.sim_x is None or self.sim_y is None:
            raise ValueError("Simulation dataset is unavailable. Generate NBx.npy and NBy.npy first.")

        if index is None:
            idx = int(np.random.randint(0, self.sim_x.shape[0]))
        else:
            idx = int(max(0, min(index, self.sim_x.shape[0] - 1)))

        return {
            "index": idx,
            "features": self.sim_x[idx].astype(np.float32).tolist(),
            "ground_truth": int(self.sim_y[idx]),
            "ground_truth_label": to_label_name(int(self.sim_y[idx])),
        }

    def simulate_attack(self, features: np.ndarray, epsilon: float) -> Dict:
        if self.uap is None:
            raise ValueError("UAP file unavailable. Run forward_project.py to generate trained_models/uap.npy.")

        eps = float(max(0.0, epsilon))
        if eps <= 0.0:
            raise ValueError("epsilon must be > 0")

        base = float(np.max(np.abs(self.uap)))
        if base <= 1e-12:
            delta = np.zeros_like(self.uap)
        else:
            delta = self.uap * (eps / base)

        clean = clamp01(features.astype(np.float32))
        adv = clamp01(clean + delta)

        pre = self.predict_one(clean)
        post = self.predict_one(adv)

        diff = adv - clean
        top_k = min(15, self.n_features)
        top_idx = np.argsort(-np.abs(diff))[:top_k]
        changed = []
        for i in top_idx:
            changed.append(
                {
                    "feature": f"f{int(i)}",
                    "before": float(clean[int(i)]),
                    "after": float(adv[int(i)]),
                    "delta": float(diff[int(i)]),
                }
            )

        return {
            "epsilon": eps,
            "before": pre,
            "after": post,
            "decision_changed": bool(pre["prediction"] != post["prediction"]),
            "top_feature_changes": changed,
        }



def create_app(service: IDSService) -> Flask:
    app = Flask(__name__, static_folder="service_ui", static_url_path="/ui")

    @app.get("/")
    def root():
        return send_from_directory("service_ui", "index.html")

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

    @app.get("/favicon.ico")
    def favicon():
        # Quiet browser favicon requests in API-only deployments.
        return ("", 204)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "time": utc_now_iso(),
                "n_features": service.n_features,
                "alert_threshold": service.alert_threshold,
            }
        )

    @app.get("/schema")
    def schema():
        return jsonify(
            {
                "n_features": service.n_features,
                "feature_keys": [f"f{i}" for i in range(service.n_features)],
                "notes": "Send normalized features in [0, 1]. Values are clipped to [0, 1] for safety.",
                "simulation_dataset_loaded": bool(service.sim_x is not None),
                "uap_loaded": bool(service.uap is not None),
            }
        )

    @app.get("/simulation_sample")
    def simulation_sample():
        try:
            idx_raw = request.args.get("index")
            index = None if idx_raw is None else int(idx_raw)
            result = service.sample_for_simulation(index)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json(force=True)
            features = coerce_features(payload, service.n_features)
            result = service.predict_one(features)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict_batch")
    def predict_batch():
        try:
            payload = request.get_json(force=True)
            features_2d = coerce_batch(payload, service.n_features)
            result = service.predict_batch(features_2d)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/simulate_attack")
    def simulate_attack():
        try:
            payload = request.get_json(force=True)
            features = coerce_features(payload, service.n_features)
            epsilon = float(payload.get("epsilon", 0.02))
            result = service.simulate_attack(features, epsilon)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/alerts")
    def get_alerts():
        limit = int(request.args.get("limit", 100))
        limit = max(1, min(limit, 500))
        return jsonify({"count": min(limit, len(service.alerts)), "alerts": service.alerts[-limit:]})

    @app.post("/predict_csv")
    def predict_csv():
        try:
            if "file" not in request.files:
                raise ValueError("Upload CSV using multipart form field named 'file'.")

            f = request.files["file"]
            if not f.filename.lower().endswith(".csv"):
                raise ValueError("Only .csv files are accepted.")

            df = pd.read_csv(f)
            features_2d = coerce_csv_dataframe(df, service.n_features)
            result = service.predict_batch(features_2d)
            result["source"] = f.filename
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-world IDS inference API service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/dqn_ids_model",
        help="Path to trained DQN model (without or with .zip)",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for raising attack alerts",
    )
    parser.add_argument(
        "--alert-log",
        type=str,
        default="logs/ids_alerts.jsonl",
        help="Path to append alert events as JSONL",
    )
    parser.add_argument(
        "--uap",
        type=str,
        default="trained_models/uap.npy",
        help="Path to UAP artifact for attack simulation",
    )
    parser.add_argument("--sim-x", type=str, default="NBx.npy", help="Path to normalized simulation features")
    parser.add_argument("--sim-y", type=str, default="NBy.npy", help="Path to simulation labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = IDSService(args.model, args.alert_threshold, args.alert_log, args.uap, args.sim_x, args.sim_y)
    app = create_app(service)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
