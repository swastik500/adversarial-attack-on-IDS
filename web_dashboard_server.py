import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import confusion_matrix
from stable_baselines3 import DQN

from forward_project import load_preprocessed_data


@dataclass
class FeatureConstraint:
    n_features: int
    mf_idx: List[int]
    rf_idx: List[int]
    uf_idx: List[int]
    recalc_rules: List[Tuple[int, int, int]]

    def apply(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone()
        if self.mf_idx:
            x_adv[self.mf_idx] = x[self.mf_idx] + delta[self.mf_idx]
        if self.uf_idx:
            x_adv[self.uf_idx] = x[self.uf_idx]
        for target_idx, num_idx, den_idx in self.recalc_rules:
            x_adv[target_idx] = x_adv[num_idx] / (x_adv[den_idx] + 1e-6)
        return torch.clamp(x_adv, 0.0, 1.0)


def build_default_groups(n_features: int):
    mf = list(range(min(20, n_features)))
    rf = list(range(min(20, n_features), min(30, n_features)))
    uf = list(range(min(30, n_features), n_features))

    recalc_rules = []
    for i, target in enumerate(rf):
        if not mf:
            break
        recalc_rules.append((target, mf[i % len(mf)], mf[(i + 1) % len(mf)]))
    return mf, rf, uf, recalc_rules


def label_name(v: int) -> str:
    return "Attack" if int(v) == 1 else "Benign"


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fn / max(tp + fn, 1))


def batch_predict_and_confidence(model: DQN, x: np.ndarray, batch_size: int = 2048):
    device = model.policy.device
    preds = []
    confs = []

    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
            q_vals = model.policy.q_net(xb)
            probs = torch.softmax(q_vals, dim=1)
            pred = torch.argmax(probs, dim=1)
            conf = torch.max(probs, dim=1).values
            preds.append(pred.cpu().numpy())
            confs.append(conf.cpu().numpy())

    return np.concatenate(preds), np.concatenate(confs)


def apply_uap_batch(x: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint, epsilon: float) -> np.ndarray:
    base_eps = float(np.max(np.abs(uap)))
    if base_eps <= 1e-12:
        delta = np.zeros_like(uap)
    else:
        delta = uap * (epsilon / base_eps)

    x_t = torch.tensor(x, dtype=torch.float32)
    d_t = torch.tensor(delta, dtype=torch.float32)
    out = torch.zeros_like(x_t)

    for i in range(x_t.shape[0]):
        out[i] = constraints.apply(x_t[i], d_t)

    return out.numpy(), delta


def compute_bundle(model: DQN, x_test: np.ndarray, y_test: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint, epsilon: float, eval_size: int):
    n = min(eval_size, x_test.shape[0])
    x_eval = x_test[:n]
    y_eval = y_test[:n]

    y_pred_before, conf_before = batch_predict_and_confidence(model, x_eval)

    x_adv, delta_scaled = apply_uap_batch(x_eval, uap, constraints, epsilon)
    y_pred_after, conf_after = batch_predict_and_confidence(model, x_adv)

    acc_before = float(np.mean(y_pred_before == y_eval))
    acc_after = float(np.mean(y_pred_after == y_eval))
    fnr_before = false_negative_rate(y_eval, y_pred_before)
    fnr_after = false_negative_rate(y_eval, y_pred_after)
    fooling_rate = float(np.mean(y_pred_before != y_pred_after))

    cm_before = confusion_matrix(y_eval, y_pred_before, labels=[0, 1]).tolist()
    cm_after = confusion_matrix(y_eval, y_pred_after, labels=[0, 1]).tolist()

    return {
        "x_eval": x_eval,
        "x_adv": x_adv,
        "y_pred_before": y_pred_before,
        "y_pred_after": y_pred_after,
        "conf_before": conf_before,
        "conf_after": conf_after,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "fnr_before": fnr_before,
        "fnr_after": fnr_after,
        "fooling_rate": fooling_rate,
        "cm_before": cm_before,
        "cm_after": cm_after,
        "delta_scaled": delta_scaled,
    }


def build_curves(model: DQN, x_test: np.ndarray, y_test: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint, eval_size: int):
    epsilons = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    rows = []
    for eps in epsilons:
        result = compute_bundle(model, x_test, y_test, uap, constraints, eps, eval_size)
        rows.append(
            {
                "epsilon": eps,
                "accuracy": result["acc_after"],
                "fnr": result["fnr_after"],
                "fooling_rate": result["fooling_rate"],
            }
        )
    return rows


app = Flask(__name__, static_folder="web")

x_train, x_test, y_train, y_test = load_preprocessed_data("NBx.npy", "NBy.npy", 0.2, 42)
model = DQN.load("trained_models/dqn_ids_model")
uap = np.load("trained_models/uap.npy").astype(np.float32)
feature_names = [f"f{i}" for i in range(x_train.shape[1])]

mf, rf, uf, recalc_rules = build_default_groups(x_train.shape[1])
constraints = FeatureConstraint(
    n_features=x_train.shape[1],
    mf_idx=mf,
    rf_idx=rf,
    uf_idx=uf,
    recalc_rules=recalc_rules,
)


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.get("/api/init")
def api_init():
    dataset_size = int(x_train.shape[0] + x_test.shape[0])
    attack_count = int(np.sum(np.concatenate([y_train, y_test]) == 1))
    benign_count = int(dataset_size - attack_count)

    sample_rows = []
    for i in range(min(10, x_train.shape[0])):
        row = {feature_names[j]: float(x_train[i][j]) for j in range(x_train.shape[1])}
        row["label_attack"] = int(y_train[i])
        sample_rows.append(row)

    try:
        with open("trained_models/forward_metrics.json", "r", encoding="utf-8") as f:
            last_metrics = json.load(f)
    except FileNotFoundError:
        last_metrics = None

    return jsonify(
        {
            "dataset_size": dataset_size,
            "n_features": int(x_train.shape[1]),
            "attack_count": attack_count,
            "benign_count": benign_count,
            "sample_rows": sample_rows,
            "feature_names": feature_names,
            "test_size": int(x_test.shape[0]),
            "last_metrics": last_metrics,
        }
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    sample_index = int(payload.get("sample_index", 0))
    sample_index = max(0, min(sample_index, x_test.shape[0] - 1))

    sample = x_test[sample_index]
    pred, conf = batch_predict_and_confidence(model, sample.reshape(1, -1))

    return jsonify(
        {
            "sample_index": sample_index,
            "sample": sample.tolist(),
            "prediction": int(pred[0]),
            "prediction_label": label_name(int(pred[0])),
            "confidence": float(conf[0]),
            "ground_truth": int(y_test[sample_index]),
            "ground_truth_label": label_name(int(y_test[sample_index])),
        }
    )


@app.post("/api/apply_attack")
def api_apply_attack():
    payload = request.get_json(force=True)
    sample_index = int(payload.get("sample_index", 0))
    sample_index = max(0, min(sample_index, x_test.shape[0] - 1))
    epsilon = float(payload.get("epsilon", 0.02))
    eval_size = int(payload.get("eval_size", 2000))
    eval_size = max(200, min(eval_size, x_test.shape[0]))

    bundle = compute_bundle(model, x_test, y_test, uap, constraints, epsilon, eval_size)
    curves = build_curves(model, x_test, y_test, uap, constraints, eval_size)

    original = bundle["x_eval"][sample_index if sample_index < bundle["x_eval"].shape[0] else 0]
    modified = bundle["x_adv"][sample_index if sample_index < bundle["x_adv"].shape[0] else 0]

    pred_before = int(bundle["y_pred_before"][sample_index if sample_index < bundle["y_pred_before"].shape[0] else 0])
    pred_after = int(bundle["y_pred_after"][sample_index if sample_index < bundle["y_pred_after"].shape[0] else 0])
    conf_before = float(bundle["conf_before"][sample_index if sample_index < bundle["conf_before"].shape[0] else 0])
    conf_after = float(bundle["conf_after"][sample_index if sample_index < bundle["conf_after"].shape[0] else 0])

    diff = modified - original
    changed_mask = np.abs(diff) > 1e-9

    feature_rows = []
    for i, fname in enumerate(feature_names):
        feature_rows.append(
            {
                "feature": fname,
                "original": float(original[i]),
                "modified": float(modified[i]),
                "delta": float(diff[i]),
                "changed": bool(changed_mask[i]),
            }
        )

    delta_scaled = bundle["delta_scaled"]
    impact = np.abs(delta_scaled)
    top_k = min(20, len(impact))
    top_idx = np.argsort(-impact)[:top_k]
    feature_impact = [
        {"feature": feature_names[int(i)], "abs_delta": float(impact[int(i)])}
        for i in top_idx
    ]

    return jsonify(
        {
            "epsilon": epsilon,
            "sample_index": sample_index,
            "feature_rows": feature_rows,
            "prediction_before": pred_before,
            "prediction_before_label": label_name(pred_before),
            "prediction_after": pred_after,
            "prediction_after_label": label_name(pred_after),
            "confidence_before": conf_before,
            "confidence_after": conf_after,
            "decision_changed": bool(pred_before != pred_after),
            "metrics": {
                "accuracy_before": float(bundle["acc_before"]),
                "accuracy_after": float(bundle["acc_after"]),
                "fnr_before": float(bundle["fnr_before"]),
                "fnr_after": float(bundle["fnr_after"]),
                "fooling_rate": float(bundle["fooling_rate"]),
            },
            "curves": curves,
            "cm_before": bundle["cm_before"],
            "cm_after": bundle["cm_after"],
            "feature_impact": feature_impact,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
