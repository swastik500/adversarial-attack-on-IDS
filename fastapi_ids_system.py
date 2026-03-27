import json
import os
import time
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


class AppState:
    def __init__(self):
        self.model = load_dqn("trained_models/dqn_ids_model")
        self.device = self.model.policy.device

        self.x_train, self.x_test, self.y_train, self.y_test = load_data("NBx.npy", "NBy.npy", 0.2, 42)
        self.n_features = int(self.x_train.shape[1])

        mf, rf, uf, recalc = build_default_groups(self.n_features)
        self.constraints = FeatureConstraint(self.n_features, mf, rf, uf, recalc)

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

    def _as_row(self, sample: List[float]) -> np.ndarray:
        x = np.asarray(sample, dtype=np.float32)
        if x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got shape={x.shape}.")
        return np.clip(x, 0.0, 1.0)

    def predict_one(self, x: np.ndarray) -> Dict:
        xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.model.policy.q_net(xt)
            probs = torch.softmax(q, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            conf = float(torch.max(probs, dim=1).values.item())
        return {"label": label_name(pred), "label_id": pred, "confidence": conf}

    def _fgsm_step(self, x: torch.Tensor, epsilon: float) -> torch.Tensor:
        x_req = x.clone().detach().requires_grad_(True)
        q = self.model.policy.q_net(x_req.unsqueeze(0))
        pred = torch.argmax(q, dim=1)
        loss = F.cross_entropy(q, pred)
        loss.backward()
        grad = x_req.grad.detach()
        x_adv = x_req + epsilon * grad.sign()
        return torch.clamp(x_adv.detach(), 0.0, 1.0)

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
        os.makedirs("trained_models", exist_ok=True)
        np.save(self.uap_path, self.uap)
        return self.uap

    def apply_attack(self, x: np.ndarray, uap_input: Optional[np.ndarray] = None) -> Dict:
        base = self.predict_one(x)

        uap = self.uap if uap_input is None else np.asarray(uap_input, dtype=np.float32)
        if uap.shape != (self.n_features,):
            raise ValueError("uap shape mismatch")

        x_adv = self.apply_uap_constraints(x.reshape(1, -1), uap)[0]
        adv = self.predict_one(x_adv)

        changed = adv["label_id"] != base["label_id"]

        # If not changed on an attack sample, adaptively scale perturbation for real effect.
        used_scale = 1.0
        if not changed and base["label_id"] == 1:
            for scale in [1.5, 2.0, 3.0, 4.0]:
                trial = np.clip(uap * scale, -0.5, 0.5)
                x_adv_trial = self.apply_uap_constraints(x.reshape(1, -1), trial)[0]
                adv_trial = self.predict_one(x_adv_trial)
                if adv_trial["label_id"] != base["label_id"]:
                    x_adv = x_adv_trial
                    adv = adv_trial
                    changed = True
                    used_scale = float(scale)
                    break

        return {
            "modified_input": x_adv.tolist(),
            "new_prediction": adv,
            "original_prediction": base,
            "prediction_changed": bool(changed),
            "used_scale": used_scale,
        }

    def metrics(self, max_eval_samples: int = 600, cache_ttl_sec: float = 300.0):
        now = time.time()
        if self._metrics_cache is not None and (now - self._metrics_cache_ts) < cache_ttl_sec:
            return self._metrics_cache

        n_total = int(self.x_test.shape[0])
        if n_total > max_eval_samples:
            idx = np.random.choice(n_total, size=max_eval_samples, replace=False)
            x_eval = self.x_test[idx]
            y_eval = self.y_test[idx]
        else:
            x_eval = self.x_test
            y_eval = self.y_test

        y_before = self.predict_batch(x_eval)
        x_adv = self.apply_uap_constraints(x_eval, self.uap)
        y_after = self.predict_batch(x_adv)

        acc_before = float(np.mean(y_before == y_eval))
        acc_after = float(np.mean(y_after == y_eval))
        fnr_before = false_negative_rate(y_eval, y_before)
        fnr_after = false_negative_rate(y_eval, y_after)
        fool = float(np.mean(y_before != y_after))

        eps_list = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        curve = []
        base = float(np.max(np.abs(self.uap)))

        for eps in eps_list:
            if base <= 1e-12:
                u = np.zeros_like(self.uap)
            else:
                u = self.uap * (eps / base)
            xa = self.apply_uap_constraints(x_eval, u)
            yp = self.predict_batch(xa)
            curve.append(
                {
                    "epsilon": eps,
                    "accuracy": float(np.mean(yp == y_eval)),
                    "fnr": false_negative_rate(y_eval, yp),
                    "fooling_rate": float(np.mean(y_before != yp)),
                }
            )

        out = {
            "accuracy_before": acc_before,
            "accuracy_after": acc_after,
            "FNR_before": fnr_before,
            "FNR_after": fnr_after,
            "fooling_rate": fool,
            "sample_size": int(x_eval.shape[0]),
            "sample_total": n_total,
            "curve": curve,
        }
        self._metrics_cache = out
        self._metrics_cache_ts = time.time()
        return out

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
app = FastAPI(title="Adversarial IDS Interactive System")
app.mount("/assets", StaticFiles(directory="fastapi_ui"), name="assets")


@app.get("/")
def home():
    return FileResponse("fastapi_ui/index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


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
        u = None if inp.uap is None else np.asarray(inp.uap, dtype=np.float32)
        out = state.apply_attack(x, u)
        return out
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/metrics")
def get_metrics():
    try:
        return state.metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_ids_system:app", host="127.0.0.1", port=9001, reload=False)
