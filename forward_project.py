import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN


def parse_index_list(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def load_preprocessed_data(x_path: str, y_path: str, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x_train, x_test, y_train, y_test


class IDSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, x: np.ndarray, y: np.ndarray):
        super().__init__()
        self.x = x
        self.y = y
        self.n_samples, self.n_features = self.x.shape
        self.index = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return self.x[self.index], {}

    def step(self, action):
        correct_label = int(self.y[self.index])
        reward = 1.0 if int(action) == correct_label else -1.0

        self.index += 1
        terminated = self.index >= self.n_samples
        truncated = False

        if terminated:
            next_obs = np.zeros((self.n_features,), dtype=np.float32)
        else:
            next_obs = self.x[self.index]

        return next_obs, reward, terminated, truncated, {}


@dataclass
class FeatureConstraint:
    n_features: int
    mf_idx: List[int]
    rf_idx: List[int]
    uf_idx: List[int]
    recalc_rules: List[Tuple[int, int, int]]

    def __post_init__(self):
        self.mf_set = set(self.mf_idx)
        self.rf_set = set(self.rf_idx)
        self.uf_set = set(self.uf_idx)

    def apply(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone()

        if self.mf_idx:
            x_adv[self.mf_idx] = x[self.mf_idx] + delta[self.mf_idx]

        if self.uf_idx:
            x_adv[self.uf_idx] = x[self.uf_idx]

        # Recompute dependent features from modifiable ones.
        for target_idx, num_idx, den_idx in self.recalc_rules:
            den = x_adv[den_idx]
            x_adv[target_idx] = x_adv[num_idx] / (den + 1e-6)

        return torch.clamp(x_adv, 0.0, 1.0)


class DQNTorchAdapter:
    def __init__(self, dqn_model: DQN, device: torch.device):
        self.model = dqn_model
        self.device = device

    def predict_label(self, x: np.ndarray) -> int:
        action, _ = self.model.predict(x, deterministic=True)
        return int(action)

    def q_values(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.policy.q_net(x_tensor)


def train_dqn_ids(x_train: np.ndarray, y_train: np.ndarray, total_timesteps: int, seed: int) -> DQN:
    env = IDSEnv(x_train, y_train)
    model = DQN(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=20000,
        learning_starts=100,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def make_balanced_training_subset(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_attack = np.where(y_train == 1)[0]
    idx_benign = np.where(y_train == 0)[0]

    if len(idx_attack) == 0 or len(idx_benign) == 0:
        return x_train, y_train

    n = min(len(idx_attack), len(idx_benign))
    attack_sample = rng.choice(idx_attack, size=n, replace=False)
    benign_sample = rng.choice(idx_benign, size=n, replace=False)
    keep = np.concatenate([attack_sample, benign_sample])
    rng.shuffle(keep)

    return x_train[keep], y_train[keep]


def fgsm(model_adapter: DQNTorchAdapter, x: torch.Tensor, epsilon: float) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    q_vals = model_adapter.q_values(x_adv.unsqueeze(0))
    pred = torch.argmax(q_vals, dim=1)

    # Maximize loss of current prediction to induce label flip.
    loss = F.cross_entropy(q_vals, pred)
    loss.backward()

    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return torch.clamp(x_adv.detach(), 0.0, 1.0)


def bim(model_adapter: DQNTorchAdapter, x: torch.Tensor, epsilon: float, alpha: float, steps: int) -> torch.Tensor:
    x_adv = x.clone().detach()
    x_orig = x.clone().detach()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        q_vals = model_adapter.q_values(x_adv.unsqueeze(0))
        pred = torch.argmax(q_vals, dim=1)
        loss = F.cross_entropy(q_vals, pred)
        loss.backward()

        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_adv.detach(), 0.0, 1.0)

    return x_adv


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_mean = a.mean()
    b_mean = b.mean()
    num = ((a - a_mean) * (b - b_mean)).sum()
    den = torch.sqrt(((a - a_mean) ** 2).sum()) * torch.sqrt(((b - b_mean) ** 2).sum())
    return num / (den + 1e-8)


def optimize_uap_with_pcc(
    model_adapter: DQNTorchAdapter,
    x: torch.Tensor,
    delta: torch.Tensor,
    constraints: FeatureConstraint,
    epsilon: float,
    pcc_steps: int,
    pcc_lr: float,
    pcc_weight: float,
) -> torch.Tensor:
    delta_opt = delta.clone().detach()

    for _ in range(pcc_steps):
        delta_opt = delta_opt.clone().detach().requires_grad_(True)
        # Keep this path fully differentiable and free of in-place index updates.
        x_adv = torch.clamp(x + delta_opt, 0.0, 1.0)

        q_adv = model_adapter.q_values(x_adv.unsqueeze(0)).squeeze(0)
        q_delta = model_adapter.q_values(torch.clamp(delta_opt, 0.0, 1.0).unsqueeze(0)).squeeze(0)

        pred_clean = model_adapter.predict_label(x.detach().cpu().numpy())
        target = torch.tensor([pred_clean], device=q_adv.device)

        ce = F.cross_entropy(q_adv.unsqueeze(0), target)
        pcc = pearson_corr(q_delta, q_adv)

        # Maximize pcc and misclassification pressure.
        objective = -ce + pcc_weight * pcc
        objective.backward()

        grad = delta_opt.grad.detach()
        delta_opt = delta_opt + pcc_lr * grad.sign()
        delta_opt = torch.clamp(delta_opt.detach(), -epsilon, epsilon)

    return delta_opt


def generate_uap(
    model_adapter: DQNTorchAdapter,
    x_train: np.ndarray,
    y_train: np.ndarray,
    constraints: FeatureConstraint,
    epsilon: float,
    attack: str,
    uap_epochs: int,
    pcc_steps: int,
    pcc_lr: float,
    pcc_weight: float,
) -> np.ndarray:
    device = model_adapter.device
    n_features = x_train.shape[1]

    uap = torch.zeros(n_features, dtype=torch.float32, device=device)

    # Focus UAP learning on attack-class samples (label=1).
    attack_indices = np.where(y_train == 1)[0]
    if len(attack_indices) == 0:
        attack_indices = np.arange(len(x_train))

    for epoch in range(uap_epochs):
        np.random.shuffle(attack_indices)
        changed = 0

        for idx in attack_indices:
            x_np = x_train[idx]
            x = torch.tensor(x_np, dtype=torch.float32, device=device)

            pred_clean = model_adapter.predict_label(x_np)
            x_uap = constraints.apply(x, uap)
            pred_uap = model_adapter.predict_label(x_uap.detach().cpu().numpy())

            if pred_clean == pred_uap:
                if attack == "bim":
                    x_adv = bim(model_adapter, x_uap, epsilon=epsilon, alpha=epsilon / 4.0, steps=5)
                else:
                    x_adv = fgsm(model_adapter, x_uap, epsilon=epsilon)

                sample_pert = x_adv - x_uap
                uap = torch.clamp(uap + sample_pert, -epsilon, epsilon)

                uap = optimize_uap_with_pcc(
                    model_adapter=model_adapter,
                    x=x,
                    delta=uap,
                    constraints=constraints,
                    epsilon=epsilon,
                    pcc_steps=pcc_steps,
                    pcc_lr=pcc_lr,
                    pcc_weight=pcc_weight,
                )
                changed += 1

        print(f"[UAP] epoch={epoch + 1}/{uap_epochs}, updated_samples={changed}")

    return uap.detach().cpu().numpy()


def predict_batch(model: DQN, x: np.ndarray) -> np.ndarray:
    preds = np.zeros((x.shape[0],), dtype=np.int64)
    for i in range(x.shape[0]):
        action, _ = model.predict(x[i], deterministic=True)
        preds[i] = int(action)
    return preds


def apply_uap_with_constraints(x: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    uap_tensor = torch.tensor(uap, dtype=torch.float32)

    out = torch.zeros_like(x_tensor)
    for i in range(x_tensor.shape[0]):
        out[i] = constraints.apply(x_tensor[i], uap_tensor)

    return out.numpy()


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fn / max(tp + fn, 1))


def evaluate_attack(model: DQN, x_test: np.ndarray, y_test: np.ndarray, x_test_adv: np.ndarray) -> dict:
    y_pred_before = predict_batch(model, x_test)
    y_pred_after = predict_batch(model, x_test_adv)

    acc_before = float(accuracy_score(y_test, y_pred_before))
    acc_after = float(accuracy_score(y_test, y_pred_after))

    fnr_before = false_negative_rate(y_test, y_pred_before)
    fnr_after = false_negative_rate(y_test, y_pred_after)

    fooling_rate = float(np.mean(y_pred_before != y_pred_after))

    metrics = {
        "accuracy_before": acc_before,
        "accuracy_after": acc_after,
        "fnr_before": fnr_before,
        "fnr_after": fnr_after,
        "fooling_rate": fooling_rate,
    }
    return metrics


def build_default_groups(n_features: int) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int, int]]]:
    # Generic default grouping when explicit feature semantics are unavailable.
    mf = list(range(min(20, n_features)))
    rf = list(range(min(20, n_features), min(30, n_features)))
    uf = list(range(min(30, n_features), n_features))

    recalc = []
    for i, target in enumerate(rf):
        num_idx = mf[i % len(mf)] if mf else 0
        den_idx = mf[(i + 1) % len(mf)] if mf else 0
        recalc.append((target, num_idx, den_idx))

    return mf, rf, uf, recalc


def main():
    parser = argparse.ArgumentParser(description="DRL IDS + UAP + Constraints + PCC pipeline")
    parser.add_argument("--x-path", default="NBx.npy")
    parser.add_argument("--y-path", default="NBy.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--uap-epochs", type=int, default=3)
    parser.add_argument("--attack", choices=["fgsm", "bim"], default="fgsm")
    parser.add_argument("--pcc-steps", type=int, default=2)
    parser.add_argument("--pcc-lr", type=float, default=0.01)
    parser.add_argument("--pcc-weight", type=float, default=0.5)
    parser.add_argument("--mf", default="")
    parser.add_argument("--rf", default="")
    parser.add_argument("--uf", default="")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_train, x_test, y_train, y_test = load_preprocessed_data(
        x_path=args.x_path,
        y_path=args.y_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Data loaded. Train={x_train.shape}, Test={x_test.shape}")

    x_train_bal, y_train_bal = make_balanced_training_subset(x_train, y_train, args.seed)
    print(f"Balanced train subset for DQN: {x_train_bal.shape}")

    dqn_model = train_dqn_ids(
        x_train=x_train_bal,
        y_train=y_train_bal,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_model.policy.to(device)
    adapter = DQNTorchAdapter(dqn_model=dqn_model, device=device)

    if args.mf or args.rf or args.uf:
        mf = parse_index_list(args.mf)
        rf = parse_index_list(args.rf)
        uf = parse_index_list(args.uf)
        recalc_rules = []
        for i, target in enumerate(rf):
            if not mf:
                break
            recalc_rules.append((target, mf[i % len(mf)], mf[(i + 1) % len(mf)]))
    else:
        mf, rf, uf, recalc_rules = build_default_groups(x_train.shape[1])

    constraints = FeatureConstraint(
        n_features=x_train.shape[1],
        mf_idx=mf,
        rf_idx=rf,
        uf_idx=uf,
        recalc_rules=recalc_rules,
    )

    uap = generate_uap(
        model_adapter=adapter,
        x_train=x_train,
        y_train=y_train,
        constraints=constraints,
        epsilon=args.epsilon,
        attack=args.attack,
        uap_epochs=args.uap_epochs,
        pcc_steps=args.pcc_steps,
        pcc_lr=args.pcc_lr,
        pcc_weight=args.pcc_weight,
    )

    x_test_adv = apply_uap_with_constraints(x_test, uap, constraints)
    metrics = evaluate_attack(dqn_model, x_test, y_test, x_test_adv)

    os.makedirs("trained_models", exist_ok=True)
    dqn_model.save("trained_models/dqn_ids_model")
    np.save("trained_models/uap.npy", uap)

    with open("trained_models/forward_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Final Metrics ===")
    print(f"Accuracy Before Attack: {metrics['accuracy_before'] * 100:.2f}%")
    print(f"Accuracy After Attack:  {metrics['accuracy_after'] * 100:.2f}%")
    print(f"FNR Before Attack:      {metrics['fnr_before'] * 100:.2f}%")
    print(f"FNR After Attack:       {metrics['fnr_after'] * 100:.2f}%")
    print(f"Fooling Rate:           {metrics['fooling_rate'] * 100:.2f}%")

    print("\nSaved artifacts:")
    print("- trained_models/dqn_ids_model.zip")
    print("- trained_models/uap.npy")
    print("- trained_models/forward_metrics.json")


if __name__ == "__main__":
    main()
