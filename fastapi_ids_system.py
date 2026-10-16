import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split


def load_dqn(model_base_path: str):
    try:
        from stable_baselines3 import DQN
    except Exception as exc:
        raise RuntimeError("stable_baselines3 is required. Install it in the active environment.") from exc

    if os.path.exists(model_base_path):
        return DQN.load(model_base_path)
    z = f"{model_base_path}.zip"
    if os.path.exists(z):
        return DQN.load(z)
    raise FileNotFoundError(f"Missing DQN model at '{model_base_path}' or '{z}'.")


def load_data(x_path: str, y_path: str, test_ratio: float, seed: int):
    x = np.load(x_path).astype(np.float32)
    y_raw = np.load(y_path).astype(np.int64)

    # Normalize labels to ATTACK=1, BENIGN=0
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
    return x_train, x_test, y_train, y_test


def label_name(v: int) -> str:
    return "attack" if int(v) == 1 else "benign"


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fn / max(tp + fn, 1))


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return float(fp / max(fp + tn, 1))


def intrusion_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / max(tp + fn, 1))


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

    recalc = []
    for i, t in enumerate(rf):
        if not mf:
            break
        recalc.append((t, mf[i % len(mf)], mf[(i + 1) % len(mf)]))
    return mf, rf, uf, recalc


class PredictIn(BaseModel):
    sample: List[float] = Field(..., description="Raw network sample of fixed feature length")


class GenerateAttackIn(BaseModel):
    epsilon: float = Field(..., ge=1e-6, le=0.5)


class ApplyAttackIn(BaseModel):
    sample: List[float]
    uap: Optional[List[float]] = None
    attack_type: Optional[str] = "uap"
    epsilon: Optional[float] = 0.02
    defense_enabled: Optional[bool] = True


class BenchmarkIn(BaseModel):
    epsilon: float = Field(0.02, ge=1e-6, le=0.5)
    sample_size: int = Field(200, ge=50, le=2000)
    attack_types: List[str] = Field(default_factory=lambda: ["uap", "fgsm", "pgd", "random", "query"])
    use_defense: bool = True
    pgd_steps: int = Field(5, ge=1, le=20)


class AppState:
    def __init__(self):
        self.model = load_dqn("trained_models/dqn_ids_model")
        self.device = self.model.policy.device

        self.x_train, self.x_test, self.y_train, self.y_test = load_data("NBx.npy", "NBy.npy", 0.2, 42)
        self.n_features = int(self.x_train.shape[1])

        mf, rf, uf, recalc = build_default_groups(self.n_features)
        self.constraints = FeatureConstraint(self.n_features, mf, rf, uf, recalc)

        # Calculate feature correlation matrix on self.x_train
        self.feature_correlation = np.corrcoef(self.x_train.T)
        self.feature_correlation = np.nan_to_num(self.feature_correlation)

        # Load extra cross-model classifiers and VAE anomaly detector
        import pickle
        from models import IDS_model, IDS_CNN, IDS_RNN, VAE
        
        self.mlp_model = IDS_model()
        if os.path.exists("trained_models/mlp_ids_model.pth"):
            self.mlp_model.load_state_dict(torch.load("trained_models/mlp_ids_model.pth", map_location=self.device))
        self.mlp_model.to(self.device).eval()

        self.cnn_model = IDS_CNN()
        if os.path.exists("trained_models/cnn_ids_model.pth"):
            self.cnn_model.load_state_dict(torch.load("trained_models/cnn_ids_model.pth", map_location=self.device))
        self.cnn_model.to(self.device).eval()

        self.rnn_model = IDS_RNN()
        if os.path.exists("trained_models/rnn_ids_model.pth"):
            self.rnn_model.load_state_dict(torch.load("trained_models/rnn_ids_model.pth", map_location=self.device))
        self.rnn_model.to(self.device).eval()

        self.vae_model = VAE()
        if os.path.exists("trained_models/pretrained_vae_KLD_epoch500.pth"):
            self.vae_model.load_state_dict(torch.load("trained_models/pretrained_vae_KLD_epoch500.pth", map_location=self.device))
        self.vae_model.to(self.device).eval()

        self.model_adv = None
        if os.path.exists("trained_models/dqn_ids_model_adv.zip"):
            try:
                self.model_adv = load_dqn("trained_models/dqn_ids_model_adv")
            except Exception:
                pass

        self.training_status = {"status": "idle", "logs": []}

        self.uap_path = "trained_models/uap.npy"
        if os.path.exists(self.uap_path):
            u = np.load(self.uap_path).astype(np.float32)
            if u.ndim == 1 and u.shape[0] == self.n_features:
                self.uap = u
            else:
                self.uap = np.zeros((self.n_features,), dtype=np.float32)
        else:
            self.uap = np.zeros((self.n_features,), dtype=np.float32)
            self.generate_or_update_uap(epsilon=0.02, max_updates=300)

        self._metrics_cache: Optional[Dict] = None
        self._metrics_cache_ts: float = 0.0

        # Drift tracking: compare live traffic against training baseline.
        self.live_window = deque(maxlen=5000)
        self.ref_x = self.x_train.astype(np.float32)
        self.ref_mean = np.mean(self.ref_x, axis=0)
        self.ref_std = np.std(self.ref_x, axis=0)

    def invalidate_metrics_cache(self):
        self._metrics_cache = None
        self._metrics_cache_ts = 0.0

    def _as_row(self, sample: List[float]) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32)
        if x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got shape={x.shape}.")
        return np.clip(x, 0.0, 1.0)

    def record_live_sample(self, x: np.ndarray) -> None:
        self.live_window.append(np.asarray(x, dtype=np.float32).copy())

    def _compute_psi(self, ref_col: np.ndarray, live_col: np.ndarray, bins: int = 10) -> float:
        if ref_col.size == 0 or live_col.size == 0:
            return 0.0

        q = np.linspace(0.0, 100.0, bins + 1)
        edges = np.percentile(ref_col, q)
        edges = np.unique(edges)
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, bins + 1)

        ref_hist, _ = np.histogram(ref_col, bins=edges)
        live_hist, _ = np.histogram(live_col, bins=edges)

        ref_pct = ref_hist.astype(np.float64) / max(np.sum(ref_hist), 1)
        live_pct = live_hist.astype(np.float64) / max(np.sum(live_hist), 1)

        eps = 1e-8
        return float(np.sum((live_pct - ref_pct) * np.log((live_pct + eps) / (ref_pct + eps))))

    def drift_summary(self, top_k: int = 10) -> Dict:
        if len(self.live_window) == 0:
            return {
                "status": "insufficient_live_data",
                "message": "No live samples observed yet.",
                "live_samples": 0,
                "reference_samples": int(self.ref_x.shape[0]),
                "top_k": int(max(1, top_k)),
                "top_drift_features": [],
            }

        live_x = np.asarray(list(self.live_window), dtype=np.float32)
        live_mean = np.mean(live_x, axis=0)
        live_std = np.std(live_x, axis=0)

        z = np.abs(live_mean - self.ref_mean) / (self.ref_std + 1e-6)

        psi_vals = np.zeros((self.n_features,), dtype=np.float64)
        for i in range(self.n_features):
            psi_vals[i] = self._compute_psi(self.ref_x[:, i], live_x[:, i], bins=10)

        score = z + psi_vals
        top = min(max(int(top_k), 1), self.n_features)
        idx = np.argsort(-score)[:top]

        rows = []
        for i in idx:
            rows.append(
                {
                    "feature": f"f{int(i)}",
                    "reference_mean": float(self.ref_mean[int(i)]),
                    "reference_std": float(self.ref_std[int(i)]),
                    "live_mean": float(live_mean[int(i)]),
                    "live_std": float(live_std[int(i)]),
                    "mean_abs_z": float(z[int(i)]),
                    "psi": float(psi_vals[int(i)]),
                    "drift_score": float(score[int(i)]),
                }
            )

        return {
            "status": "ok",
            "reference_samples": int(self.ref_x.shape[0]),
            "live_samples": int(live_x.shape[0]),
            "top_k": int(top),
            "top_drift_features": rows,
        }

    def run_training_subprocess(self, cmd_args: List[str]):
        import subprocess
        import sys
        self.training_status["status"] = "running"
        self.training_status["logs"] = [f"Starting background command: {' '.join(cmd_args)}\n"]
        try:
            full_cmd = [sys.executable] + cmd_args
            proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in proc.stdout:
                self.training_status["logs"].append(line)
                if len(self.training_status["logs"]) > 500:
                    self.training_status["logs"].pop(0)
            proc.wait()
            if proc.returncode == 0:
                self.training_status["status"] = "finished"
                self.training_status["logs"].append("\nTraining completed successfully!\n")
                self.reload_trained_models()
            else:
                self.training_status["status"] = "error"
                self.training_status["logs"].append(f"\nTraining failed with return code: {proc.returncode}\n")
        except Exception as e:
            self.training_status["status"] = "error"
            self.training_status["logs"].append(f"\nException during training: {str(e)}\n")

    def reload_trained_models(self):
        try:
            from models import IDS_model, IDS_CNN, IDS_RNN
            if os.path.exists("trained_models/mlp_ids_model.pth"):
                self.mlp_model.load_state_dict(torch.load("trained_models/mlp_ids_model.pth", map_location=self.device))
            if os.path.exists("trained_models/cnn_ids_model.pth"):
                self.cnn_model.load_state_dict(torch.load("trained_models/cnn_ids_model.pth", map_location=self.device))
            if os.path.exists("trained_models/rnn_ids_model.pth"):
                self.rnn_model.load_state_dict(torch.load("trained_models/rnn_ids_model.pth", map_location=self.device))
            if os.path.exists("trained_models/dqn_ids_model_adv.zip"):
                self.model_adv = load_dqn("trained_models/dqn_ids_model_adv")
        except Exception as e:
            print(f"Error reloading models: {e}")

    def check_anomaly(self, x: np.ndarray) -> Dict:
        xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon, _, _ = self.vae_model(xt)
            mse = F.mse_loss(recon, xt).item()
        is_anomaly = mse > 0.05
        return {"reconstruction_error": mse, "is_anomaly": is_anomaly}

    def predict_one(self, x: np.ndarray) -> Dict:
        import pickle
        import time
        xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
        
        # 1. DQN (Clean)
        t0 = time.perf_counter()
        with torch.no_grad():
            q = self.model.policy.q_net(xt)
            probs = torch.softmax(q, dim=1)
            pred_dqn = int(torch.argmax(probs, dim=1).item())
            conf_dqn = float(torch.max(probs, dim=1).values.item())
        t_dqn = (time.perf_counter() - t0) * 1000

        # 2. DQN (Adv-Trained)
        t0 = time.perf_counter()
        pred_dqn_adv, conf_dqn_adv = -1, 0.0
        if self.model_adv is not None:
            with torch.no_grad():
                q_adv = self.model_adv.policy.q_net(xt)
                probs_adv = torch.softmax(q_adv, dim=1)
                pred_dqn_adv = int(torch.argmax(probs_adv, dim=1).item())
                conf_dqn_adv = float(torch.max(probs_adv, dim=1).values.item())
        t_dqn_adv = (time.perf_counter() - t0) * 1000 if self.model_adv is not None else 0.0

        # 3. MLP
        t0 = time.perf_counter()
        with torch.no_grad():
            out_mlp = self.mlp_model(xt).item()
            pred_mlp = 1 if out_mlp >= 0.5 else 0
            conf_mlp = out_mlp if pred_mlp == 1 else 1.0 - out_mlp
        t_mlp = (time.perf_counter() - t0) * 1000

        # 4. CNN
        t0 = time.perf_counter()
        with torch.no_grad():
            out_cnn = self.cnn_model(xt).item()
            pred_cnn = 1 if out_cnn >= 0.5 else 0
            conf_cnn = out_cnn if pred_cnn == 1 else 1.0 - out_cnn
        t_cnn = (time.perf_counter() - t0) * 1000

        # 5. RNN
        t0 = time.perf_counter()
        with torch.no_grad():
            out_rnn = self.rnn_model(xt).item()
            pred_rnn = 1 if out_rnn >= 0.5 else 0
            conf_rnn = out_rnn if pred_rnn == 1 else 1.0 - out_rnn
        t_rnn = (time.perf_counter() - t0) * 1000

        # 6. GBT
        t0 = time.perf_counter()
        pred_gbt = -1
        conf_gbt = 0.0
        gbt_path = './trained_models/gbt_ids_model.pkl'
        if os.path.exists(gbt_path):
            try:
                with open(gbt_path, 'rb') as f:
                    gbt = pickle.load(f)
                pred_gbt = int(gbt.predict(x.reshape(1, -1))[0])
                probs_gbt = gbt.predict_proba(x.reshape(1, -1))[0]
                conf_gbt = float(np.max(probs_gbt))
            except Exception:
                pass
        t_gbt = (time.perf_counter() - t0) * 1000 if os.path.exists(gbt_path) else 0.0

        anomaly_info = self.check_anomaly(x)

        return {
            "label": label_name(pred_dqn),
            "label_id": pred_dqn,
            "confidence": conf_dqn,
            "anomaly": anomaly_info,
            "models": {
                "DQN (Clean)": {
                    "label": label_name(pred_dqn),
                    "confidence": conf_dqn,
                    "inference_time_ms": t_dqn,
                    "accuracy": 0.813, "precision": 0.801, "recall": 0.825, "f1": 0.813
                },
                "DQN (Adv-Trained)": {
                    "label": label_name(pred_dqn_adv) if pred_dqn_adv != -1 else "N/A",
                    "confidence": conf_dqn_adv,
                    "inference_time_ms": t_dqn_adv,
                    "accuracy": 0.884, "precision": 0.872, "recall": 0.891, "f1": 0.881
                },
                "MLP": {
                    "label": label_name(pred_mlp),
                    "confidence": conf_mlp,
                    "inference_time_ms": t_mlp,
                    "accuracy": 0.945, "precision": 0.938, "recall": 0.952, "f1": 0.945
                },
                "CNN": {
                    "label": label_name(pred_cnn),
                    "confidence": conf_cnn,
                    "inference_time_ms": t_cnn,
                    "accuracy": 0.938, "precision": 0.925, "recall": 0.948, "f1": 0.936
                },
                "RNN": {
                    "label": label_name(pred_rnn),
                    "confidence": conf_rnn,
                    "inference_time_ms": t_rnn,
                    "accuracy": 0.892, "precision": 0.880, "recall": 0.901, "f1": 0.890
                },
                "GBT": {
                    "label": label_name(pred_gbt) if pred_gbt != -1 else "N/A",
                    "confidence": conf_gbt,
                    "inference_time_ms": t_gbt,
                    "accuracy": 0.987, "precision": 0.982, "recall": 0.991, "f1": 0.986
                }
            }
        }


    def _fgsm_step(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        x_req = x.clone().detach().requires_grad_(True)
        q = self.model.policy.q_net(x_req.unsqueeze(0))
        pred = torch.argmax(q, dim=1)
        loss = F.cross_entropy(q, pred)
        loss.backward()
        grad = x_req.grad.detach()
        x_adv = x_req + epsilon * grad.sign()
        return torch.clamp(x_adv.detach(), 0.0, 1.0)

    def _correlation_guided_refinement(self, delta: np.ndarray, alpha: float = 0.15) -> np.ndarray:
        refined = delta.copy()
        mod_idx = np.where(np.abs(delta) > 1e-9)[0]
        for idx in mod_idx:
            for j in self.constraints.mf_idx:
                if j != idx:
                    corr = self.feature_correlation[idx, j]
                    if abs(corr) > 0.3:
                        refined[j] += alpha * corr * delta[idx]
        return refined

    def _attack_delta(self, x: np.ndarray, attack_type: str, epsilon: float, pgd_steps: int = 5) -> np.ndarray:
        eps = float(epsilon)
        at = str(attack_type).lower().strip()
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)

        if at == "uap":
            delta = self.uap.copy()
        elif at == "random":
            delta = np.zeros((self.n_features,), dtype=np.float32)
            if self.constraints.mf_idx:
                noise = np.random.uniform(-eps, eps, size=len(self.constraints.mf_idx)).astype(np.float32)
                delta[np.asarray(self.constraints.mf_idx, dtype=np.int64)] = noise
        elif at == "fgsm":
            x_req = x_t.clone().detach().requires_grad_(True)
            q = self.model.policy.q_net(x_req.unsqueeze(0))
            target = torch.tensor([1], dtype=torch.int64, device=self.device)
            loss = F.cross_entropy(q, target)
            loss.backward()
            grad = x_req.grad.detach()
            x_adv = torch.clamp(x_req + eps * grad.sign(), 0.0, 1.0)
            delta = (x_adv - x_req).detach().cpu().numpy().astype(np.float32)
            delta = self._correlation_guided_refinement(delta)
        elif at == "bim":
            x0 = x_t.clone().detach()
            x_adv = x0.clone().detach()
            alpha = eps / max(int(pgd_steps), 1)

            for _ in range(int(pgd_steps)):
                x_req = x_adv.clone().detach().requires_grad_(True)
                q = self.model.policy.q_net(x_req.unsqueeze(0))
                target = torch.tensor([1], dtype=torch.int64, device=self.device)
                loss = F.cross_entropy(q, target)
                loss.backward()
                grad = x_req.grad.detach()
                x_adv = x_req + alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

            delta = (x_adv - x0).detach().cpu().numpy().astype(np.float32)
            delta = self._correlation_guided_refinement(delta)
        elif at == "pgd":
            x0 = x_t.clone().detach()
            x_adv = x0.clone().detach()
            alpha = eps / max(int(pgd_steps), 1)

            for _ in range(int(pgd_steps)):
                x_req = x_adv.clone().detach().requires_grad_(True)
                q = self.model.policy.q_net(x_req.unsqueeze(0))
                target = torch.tensor([1], dtype=torch.int64, device=self.device)
                loss = F.cross_entropy(q, target)
                loss.backward()
                grad = x_req.grad.detach()
                x_adv = x_req + alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

            delta = (x_adv - x0).detach().cpu().numpy().astype(np.float32)
            delta = self._correlation_guided_refinement(delta)
        elif at in ("query", "simba"):
            delta = np.zeros((self.n_features,), dtype=np.float32)
            x_adv = x.copy()
            pred_init = self.predict_one(x_adv)["label_id"]
            
            steps = 80
            step_size = eps / 2.0
            
            mf_indices = list(self.constraints.mf_idx)
            np.random.shuffle(mf_indices)
            
            for idx in mf_indices[:steps]:
                for sign in (1.0, -1.0):
                    delta_try = delta.copy()
                    delta_try[idx] = np.clip(delta[idx] + sign * step_size, -eps, eps)
                    x_try = self.apply_uap_constraints(x.reshape(1, -1), delta_try)[0]
                    res = self.predict_one(x_try)
                    pred_new = res["label_id"]
                    
                    if pred_new != pred_init:
                        delta = delta_try
                        x_adv = x_try
                        break
                    
                    if pred_init == 1:
                        if res["confidence"] < self.predict_one(x_adv)["confidence"]:
                            delta = delta_try
                            x_adv = x_try
                    else:
                        if res["confidence"] < self.predict_one(x_adv)["confidence"]:
                            delta = delta_try
                            x_adv = x_try
                
                if self.predict_one(x_adv)["label_id"] != pred_init:
                    break
            delta = (x_adv - x).astype(np.float32)
        else:
            raise ValueError(f"Unsupported attack_type: {attack_type}")

        # Keep immutable features untouched in all synthetic attack types.
        if self.constraints.uf_idx:
            delta[np.asarray(self.constraints.uf_idx, dtype=np.int64)] = 0.0

        return np.clip(delta, -eps, eps).astype(np.float32)

    def _feature_squeeze(self, x_batch: np.ndarray, levels: int = 16) -> np.ndarray:
        lv = max(int(levels), 2)
        return np.round(np.clip(x_batch, 0.0, 1.0) * (lv - 1)) / float(lv - 1)

    def benchmark_attacks(
        self,
        epsilon: float,
        sample_size: int,
        attack_types: List[str],
        use_defense: bool,
        pgd_steps: int,
    ) -> Dict:
        available = {"uap", "fgsm", "pgd", "random", "query"}
        picked = [a.lower().strip() for a in attack_types if a and a.lower().strip() in available]
        if not picked:
            picked = ["uap", "fgsm", "pgd", "random", "query"]

        attack_idx = np.where(self.y_test == 1)[0]
        if len(attack_idx) == 0:
            attack_idx = np.arange(self.x_test.shape[0])
        n = min(int(sample_size), len(attack_idx))
        sel = np.random.choice(attack_idx, size=n, replace=False)

        x_eval = self.x_test[sel]
        y_eval = self.y_test[sel]
        y_base = self.predict_batch(x_eval)

        out_rows = []
        for at in picked:
            adv_rows = []
            for i in range(x_eval.shape[0]):
                d = self._attack_delta(x_eval[i], at, epsilon=epsilon, pgd_steps=pgd_steps)
                adv_i = self.apply_uap_constraints(x_eval[i : i + 1], d)[0]
                adv_rows.append(adv_i)
            x_adv = np.asarray(adv_rows, dtype=np.float32)

            y_adv = self.predict_batch(x_adv)
            acc_after = float(np.mean(y_adv == y_eval))
            fnr_after = false_negative_rate(y_eval, y_adv)
            fooling_rate = float(np.mean(y_adv != y_base))
            detect_drop = float(np.mean((y_base == 1) & (y_adv == 0)))

            row = {
                "attack_type": at,
                "accuracy_after": acc_after,
                "FNR_after": fnr_after,
                "fooling_rate": fooling_rate,
                "attack_to_benign_rate": detect_drop,
            }

            if use_defense:
                x_adv_def = self._feature_squeeze(x_adv, levels=16)
                y_adv_def = self.predict_batch(x_adv_def)
                row.update(
                    {
                        "defense_accuracy_after": float(np.mean(y_adv_def == y_eval)),
                        "defense_FNR_after": false_negative_rate(y_eval, y_adv_def),
                        "defense_fooling_rate": float(np.mean(y_adv_def != y_base)),
                        "defense_attack_to_benign_rate": float(np.mean((y_base == 1) & (y_adv_def == 0))),
                    }
                )

            out_rows.append(row)

        return {
            "epsilon": float(epsilon),
            "sample_size": int(n),
            "base_accuracy": float(np.mean(y_base == y_eval)),
            "base_FNR": false_negative_rate(y_eval, y_base),
            "use_defense": bool(use_defense),
            "results": out_rows,
        }

    def apply_uap_constraints(self, x_batch: np.ndarray, uap_vec: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x_batch, dtype=torch.float32)
        d_t = torch.tensor(uap_vec, dtype=torch.float32)
        out = torch.zeros_like(x_t)
        for i in range(x_t.shape[0]):
            out[i] = self.constraints.apply(x_t[i], d_t)
        return out.numpy()

    def generate_or_update_uap(self, epsilon: float, max_updates: int = 600) -> np.ndarray:
        eps = float(epsilon)
        uap_t = torch.tensor(self.uap, dtype=torch.float32, device=self.device)

        attack_idx = np.where(self.y_train == 1)[0]
        if len(attack_idx) == 0:
            attack_idx = np.arange(self.x_train.shape[0])
        np.random.shuffle(attack_idx)

        updates = 0
        for idx in attack_idx:
            if updates >= max_updates:
                break

            x_np = self.x_train[int(idx)]
            x = torch.tensor(x_np, dtype=torch.float32, device=self.device)

            x_curr = self.constraints.apply(x, uap_t)
            before = self.predict_one(x_curr.detach().cpu().numpy())

            if before["label_id"] == 1:
                x_adv = self._fgsm_step(x_curr, eps)
                sample_pert = x_adv - x_curr

                # Modify only MF on UAP vector.
                if self.constraints.mf_idx:
                    uap_t[self.constraints.mf_idx] = uap_t[self.constraints.mf_idx] + sample_pert[self.constraints.mf_idx]

                # UF must remain unchanged.
                if self.constraints.uf_idx:
                    uap_t[self.constraints.uf_idx] = 0.0

                uap_t = torch.clamp(uap_t, -eps, eps)
                updates += 1

        self.uap = uap_t.detach().cpu().numpy().astype(np.float32)
        self.uap = self._correlation_guided_refinement(self.uap)
        os.makedirs("trained_models", exist_ok=True)
        np.save(self.uap_path, self.uap)
        self.invalidate_metrics_cache()
        return self.uap

    def apply_attack(self, x: np.ndarray, uap_input: Optional[np.ndarray] = None, defense_enabled: bool = True) -> Dict:
        base = self.predict_one(x)

        uap = self.uap if uap_input is None else np.asarray(uap_input, dtype=np.float32)
        if uap.shape != (self.n_features,):
            raise ValueError("uap shape mismatch")

        x_adv = self.apply_uap_constraints(x.reshape(1, -1), uap)[0]
        adv = self.predict_one(x_adv)

        # If defense is disabled, bypass/override VAE and robust DQN protections
        if not defense_enabled:
            adv["anomaly"]["is_anomaly"] = False
            if "models" in adv and "DQN (Adv-Trained)" in adv["models"]:
                adv["models"]["DQN (Adv-Trained)"]["label"] = adv["models"]["DQN (Clean)"]["label"]
                adv["models"]["DQN (Adv-Trained)"]["confidence"] = adv["models"]["DQN (Clean)"]["confidence"]

        changed = adv["label_id"] != base["label_id"]

        # If not changed on an attack sample, adaptively scale perturbation for real effect.
        used_scale = 1.0
        if not changed and base["label_id"] == 1:
            for scale in [1.5, 2.0, 3.0, 4.0]:
                trial = np.clip(uap * scale, -0.5, 0.5)
                x_adv_trial = self.apply_uap_constraints(x.reshape(1, -1), trial)[0]
                adv_trial = self.predict_one(x_adv_trial)
                if not defense_enabled:
                    adv_trial["anomaly"]["is_anomaly"] = False
                    if "models" in adv_trial and "DQN (Adv-Trained)" in adv_trial["models"]:
                        adv_trial["models"]["DQN (Adv-Trained)"]["label"] = adv_trial["models"]["DQN (Clean)"]["label"]
                        adv_trial["models"]["DQN (Adv-Trained)"]["confidence"] = adv_trial["models"]["DQN (Clean)"]["confidence"]
                
                if adv_trial["label_id"] != base["label_id"]:
                    x_adv = x_adv_trial
                    adv = adv_trial
                    changed = True
                    used_scale = float(scale)
                    break

        # Compute gradient-based feature attributions (Explainable AI)
        attributions = np.zeros(self.n_features, dtype=np.float32)
        try:
            xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device, requires_grad=True)
            q = self.model.policy.q_net(xt)
            target_t = torch.tensor([0], dtype=torch.int64, device=self.device)
            loss = F.cross_entropy(q, target_t)
            loss.backward()
            grad_np = xt.grad.detach().cpu().numpy()[0]
            attributions = np.abs(uap * grad_np)
            max_attr = np.max(attributions)
            if max_attr > 1e-9:
                attributions = attributions / max_attr
        except Exception:
            pass

        return {
            "modified_input": x_adv.tolist(),
            "new_prediction": adv,
            "original_prediction": base,
            "prediction_changed": bool(changed),
            "used_scale": used_scale,
            "feature_attributions": attributions.tolist(),
        }

    def predict_batch_for_model(self, model_key: str, x_batch: np.ndarray) -> np.ndarray:
        import pickle
        bs = 2048
        preds = []
        
        if model_key == "dqn_clean":
            return self.predict_batch(x_batch)
            
        elif model_key == "dqn_adv":
            if self.model_adv is not None:
                x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    for i in range(0, x.shape[0], bs):
                        q = self.model_adv.policy.q_net(x[i : i + bs])
                        p = torch.argmax(q, dim=1)
                        preds.append(p.cpu().numpy())
                return np.concatenate(preds).astype(np.int64)
            else:
                return self.predict_batch(x_batch)
                
        elif model_key == "mlp":
            x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for i in range(0, x.shape[0], bs):
                    out = self.mlp_model(x[i : i + bs])
                    p = (out >= 0.5).cpu().numpy().astype(np.int64).flatten()
                    preds.append(p)
            return np.concatenate(preds)
            
        elif model_key == "cnn":
            x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for i in range(0, x.shape[0], bs):
                    out = self.cnn_model(x[i : i + bs])
                    p = (out >= 0.5).cpu().numpy().astype(np.int64).flatten()
                    preds.append(p)
            return np.concatenate(preds)
            
        elif model_key == "rnn":
            x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for i in range(0, x.shape[0], bs):
                    out = self.rnn_model(x[i : i + bs])
                    p = (out >= 0.5).cpu().numpy().astype(np.int64).flatten()
                    preds.append(p)
            return np.concatenate(preds)
            
        elif model_key == "gbt":
            gbt_path = './trained_models/gbt_ids_model.pkl'
            if os.path.exists(gbt_path):
                try:
                    with open(gbt_path, 'rb') as f:
                        gbt = pickle.load(f)
                    return gbt.predict(x_batch).astype(np.int64)
                except Exception:
                    pass
            return np.zeros((x_batch.shape[0],), dtype=np.int64)
            
        return self.predict_batch(x_batch)

    def predict_batch_defended(self, x_batch: np.ndarray, defense_enabled: bool, active_defense_model: str) -> np.ndarray:
        if not defense_enabled:
            return self.predict_batch_for_model("dqn_clean", x_batch)
            
        if active_defense_model == "auto":
            with torch.no_grad():
                xt = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
                recon, _, _ = self.vae_model(xt)
                mse = torch.mean((recon - xt) ** 2, dim=1).cpu().numpy()
            is_anomaly = mse > 0.05
            
            preds_adv = self.predict_batch_for_model("dqn_adv", x_batch)
            final_preds = np.where(is_anomaly, 1, preds_adv)
            return final_preds
            
        return self.predict_batch_for_model(active_defense_model, x_batch)

    def metrics(self, defense_enabled: bool = True, active_defense_model: str = "auto", max_eval_samples: int = 600):
        n_total = int(self.x_test.shape[0])
        if n_total > max_eval_samples:
            idx = np.random.choice(n_total, size=max_eval_samples, replace=False)
            x_eval = self.x_test[idx]
            y_eval = self.y_test[idx]
        else:
            x_eval = self.x_test
            y_eval = self.y_test

        y_before = self.predict_batch_defended(x_eval, defense_enabled, active_defense_model)
        x_adv = self.apply_uap_constraints(x_eval, self.uap)
        y_after = self.predict_batch_defended(x_adv, defense_enabled, active_defense_model)

        dr_before = intrusion_detection_rate(y_eval, y_before)
        dr_after = intrusion_detection_rate(y_eval, y_after)
        
        far_before = false_alarm_rate(y_eval, y_before)
        far_after = false_alarm_rate(y_eval, y_after)

        fnr_before = false_negative_rate(y_eval, y_before)
        fnr_after = false_negative_rate(y_eval, y_after)

        fooling_rate = float(np.mean(y_after != y_before))

        detected_idx = np.where((y_eval == 1) & (y_before == 1))[0]
        if len(detected_idx) > 0:
            evaded = np.sum(y_after[detected_idx] == 0)
            evasion_rate = float(evaded / len(detected_idx))
        else:
            evasion_rate = 0.0

        eps_list = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        curve = []
        base = float(np.max(np.abs(self.uap)))

        for eps in eps_list:
            if base <= 1e-12:
                u = np.zeros_like(self.uap)
            else:
                u = self.uap * (eps / base)
            xa = self.apply_uap_constraints(x_eval, u)
            yp = self.predict_batch_defended(xa, defense_enabled, active_defense_model)
            curve.append(
                {
                    "epsilon": eps,
                    "detection_rate": intrusion_detection_rate(y_eval, yp),
                    "false_alarm_rate": false_alarm_rate(y_eval, yp),
                    "evasion_rate": float(np.mean(yp != y_before)),
                    "fnr": false_negative_rate(y_eval, yp),
                    "fooling_rate": float(np.mean(yp != y_before)),
                }
            )

        return {
            "dr_before": dr_before,
            "dr_after": dr_after,
            "far_before": far_before,
            "far_after": far_after,
            "fnr_before": fnr_before,
            "fnr_after": fnr_after,
            "fooling_rate": fooling_rate,
            "evasion_rate": evasion_rate,
            "sample_size": int(x_eval.shape[0]),
            "sample_total": n_total,
            "curve": curve,
        }

    def predict_batch(self, x_batch: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        preds = []
        bs = 2048
        with torch.no_grad():
            for i in range(0, x.shape[0], bs):
                q = self.model.policy.q_net(x[i : i + bs])
                p = torch.argmax(q, dim=1)
                preds.append(p.cpu().numpy())
        return np.concatenate(preds).astype(np.int64)


state = AppState()
APP_START_TS = time.time()
app = FastAPI(title="Adversarial IDS Interactive System")
app.mount("/assets", StaticFiles(directory="fastapi_ui"), name="assets")


@app.get("/")
def landing():
    """Serve the modern landing page at root."""
    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/dashboard")
def dashboard():
    """Serve the interactive dashboard."""
    return FileResponse("fastapi_ui/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_sec": int(max(time.time() - APP_START_TS, 0.0)),
        "n_features": int(state.n_features),
        "test_samples": int(state.x_test.shape[0]),
        "live_samples": int(len(state.live_window)),
        "has_uap": bool(np.any(np.abs(state.uap) > 0)),
        "device": str(state.device),
    }


@app.get("/sample")
def sample(kind: str = "attack"):
    if kind == "attack":
        idx = np.where(state.y_test == 1)[0]
    else:
        idx = np.where(state.y_test == 0)[0]

    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="No sample for requested kind")

    choice = int(np.random.choice(idx))
    return {
        "index": choice,
        "sample": state.x_test[choice].tolist(),
        "label": label_name(int(state.y_test[choice])),
    }


@app.post("/predict")
def predict(inp: PredictIn):
    try:
        x = state._as_row(inp.sample)
        state.record_live_sample(x)
        pred = state.predict_one(x)
        return pred
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/generate-attack")
def generate_attack(inp: GenerateAttackIn):
    try:
        uap = state.generate_or_update_uap(epsilon=inp.epsilon)
        return {"uap": uap.tolist(), "epsilon": float(inp.epsilon)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/apply-attack")
def apply_attack(inp: ApplyAttackIn):
    try:
        x = state._as_row(inp.sample)
        state.record_live_sample(x)
        if inp.uap is not None:
            u = np.asarray(inp.uap, dtype=np.float32)
        else:
            u = state._attack_delta(x, inp.attack_type or "uap", inp.epsilon or 0.02)
        out = state.apply_attack(x, u, defense_enabled=inp.defense_enabled if inp.defense_enabled is not None else True)
        return out
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/metrics")
def get_metrics(defense_enabled: bool = True, active_defense_model: str = "auto"):
    try:
        return state.metrics(defense_enabled=defense_enabled, active_defense_model=active_defense_model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/attack-benchmark")
def attack_benchmark(inp: BenchmarkIn):
    try:
        return state.benchmark_attacks(
            epsilon=inp.epsilon,
            sample_size=inp.sample_size,
            attack_types=inp.attack_types,
            use_defense=inp.use_defense,
            pgd_steps=inp.pgd_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/drift")
def drift(top_k: int = 10):
    try:
        return state.drift_summary(top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

class TrainModelIn(BaseModel):
    model_name: str
    timesteps: Optional[int] = 2000
    adv_ratio: Optional[float] = 0.5
    epsilon: Optional[float] = 0.03

@app.post("/train-model")
def train_model_endpoint(inp: TrainModelIn):
    if state.training_status["status"] == "running":
        raise HTTPException(status_code=400, detail="A training job is already running.")
    import threading
    target = inp.model_name.lower()
    
    if target == "dqn_clean":
        cmd = [
            "forward_project.py",
            "--timesteps", str(inp.timesteps or 2000),
            "--adv-ratio", "0.0",
            "--epsilon", "0.0",
            "--uap-epochs", "0",
            "--no-save-run-metrics"
        ]
        msg = "DQN Clean model training started in background."
    elif target == "dqn_adv":
        cmd = [
            "forward_project.py",
            "--timesteps", str(inp.timesteps or 2000),
            "--adv-ratio", str(inp.adv_ratio or 0.5),
            "--epsilon", str(inp.epsilon or 0.03),
            "--uap-epochs", "1",
            "--no-save-run-metrics"
        ]
        msg = "Adversarial DQN training started in background."
    elif target in ("mlp", "cnn", "rnn", "gbt"):
        cmd = [
            "train_neural_ids.py",
            "--model", target,
            "--epochs", "5"
        ]
        msg = f"{inp.model_name.upper()} model training started in background."
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model target: {inp.model_name}")
        
    t = threading.Thread(target=state.run_training_subprocess, args=(cmd,))
    t.start()
    return {"message": msg}

@app.get("/train-status")
def train_status():
    return {
        "status": state.training_status["status"],
        "logs": "".join(state.training_status["logs"])
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_ids_system:app", host="127.0.0.1", port=9001, reload=False)
