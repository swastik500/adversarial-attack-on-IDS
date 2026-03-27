import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.metrics import confusion_matrix
from stable_baselines3 import DQN

from forward_project import load_preprocessed_data


st.set_page_config(page_title="ThreatShield Dashboard", layout="wide")


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


def build_default_groups(n_features: int) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int, int]]]:
    mf = list(range(min(20, n_features)))
    rf = list(range(min(20, n_features), min(30, n_features)))
    uf = list(range(min(30, n_features), n_features))

    recalc_rules = []
    for i, target in enumerate(rf):
        if not mf:
            break
        recalc_rules.append((target, mf[i % len(mf)], mf[(i + 1) % len(mf)]))
    return mf, rf, uf, recalc_rules


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_train, x_test, y_train, y_test = load_preprocessed_data(
        x_path="NBx.npy",
        y_path="NBy.npy",
        test_ratio=0.2,
        seed=42,
    )
    feature_names = [f"f{i}" for i in range(x_train.shape[1])]
    sample_df = pd.DataFrame(x_train[:10], columns=feature_names)
    sample_df["label_attack"] = y_train[:10]
    return x_train, x_test, y_train, y_test, sample_df


@st.cache_resource(show_spinner=False)
def load_model() -> DQN:
    return DQN.load("trained_models/dqn_ids_model")


@st.cache_data(show_spinner=False)
def load_uap() -> np.ndarray:
    return np.load("trained_models/uap.npy").astype(np.float32)


def batch_predict_and_confidence(model: DQN, x: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
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


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fn / max(tp + fn, 1))


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

    return out.numpy()


def compute_bundle(model: DQN, x_test: np.ndarray, y_test: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint, epsilon: float, eval_size: int):
    n = min(eval_size, x_test.shape[0])
    x_eval = x_test[:n]
    y_eval = y_test[:n]

    y_pred_before, conf_before = batch_predict_and_confidence(model, x_eval)

    x_adv = apply_uap_batch(x_eval, uap, constraints, epsilon)
    y_pred_after, conf_after = batch_predict_and_confidence(model, x_adv)

    acc_before = float(np.mean(y_pred_before == y_eval))
    acc_after = float(np.mean(y_pred_after == y_eval))
    fnr_before = false_negative_rate(y_eval, y_pred_before)
    fnr_after = false_negative_rate(y_eval, y_pred_after)
    fooling_rate = float(np.mean(y_pred_before != y_pred_after))

    cm_before = confusion_matrix(y_eval, y_pred_before, labels=[0, 1])
    cm_after = confusion_matrix(y_eval, y_pred_after, labels=[0, 1])

    return {
        "x_eval": x_eval,
        "x_adv": x_adv,
        "y_eval": y_eval,
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
    }


def build_curves(model: DQN, x_test: np.ndarray, y_test: np.ndarray, uap: np.ndarray, constraints: FeatureConstraint, eval_size: int):
    epsilons = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)
    rows = []
    for eps in epsilons:
        result = compute_bundle(model, x_test, y_test, uap, constraints, float(eps), eval_size)
        rows.append(
            {
                "epsilon": float(eps),
                "accuracy": result["acc_after"],
                "fnr": result["fnr_after"],
                "fooling_rate": result["fooling_rate"],
            }
        )
    return pd.DataFrame(rows)


def label_name(v: int) -> str:
    return "Attack" if int(v) == 1 else "Benign"


def card(title: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div style='padding:14px;border:1px solid #ddd;border-radius:12px;background:#fafafa;'>
            <div style='font-size:0.85rem;color:#555;'>{title}</div>
            <div style='font-size:1.6rem;font-weight:700;margin-top:4px;'>{value}</div>
            <div style='font-size:0.8rem;color:#666;margin-top:4px;'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("ThreatShield – Adversarial Attack on IDS")
st.write("Demonstrating how small perturbations can bypass AI security systems")

x_train, x_test, y_train, y_test, sample_df = load_data()
model = load_model()
uap = load_uap()

mf, rf, uf, recalc_rules = build_default_groups(x_train.shape[1])
constraints = FeatureConstraint(
    n_features=x_train.shape[1],
    mf_idx=mf,
    rf_idx=rf,
    uf_idx=uf,
    recalc_rules=recalc_rules,
)

st.sidebar.header("Simulation Controls")
epsilon = st.sidebar.slider("Attack strength (epsilon)", min_value=0.0, max_value=0.05, value=0.02, step=0.005)
eval_size = st.sidebar.slider("Evaluation sample size", min_value=500, max_value=6000, value=2000, step=500)
row_index = st.sidebar.slider("Input sample index", min_value=0, max_value=min(5000, x_test.shape[0] - 1), value=0, step=1)

col_a, col_b = st.columns([1.1, 1.0])
with col_a:
    st.subheader("1) Dataset Panel")
    dataset_size = x_train.shape[0] + x_test.shape[0]
    attack_count = int(np.sum(np.concatenate([y_train, y_test]) == 1))
    benign_count = dataset_size - attack_count

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Dataset Size", f"{dataset_size:,}", "Train + Test samples")
    with c2:
        card("Features", f"{x_train.shape[1]}", "Per network flow")
    with c3:
        card("Class Distribution", f"A:{attack_count:,} / B:{benign_count:,}", "Attack vs Benign")

    st.write("Sample network rows")
    st.dataframe(sample_df, use_container_width=True)

with col_b:
    st.subheader("10) Workflow Diagram")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box style=filled fillcolor="#eef5ff"];
            dataset [label="Dataset"];
            drl [label="DRL Model (DQN)"];
            uap [label="UAP Generator"];
            modified [label="Modified Input"];
            pred [label="Prediction"];
            dataset -> drl -> uap -> modified -> pred;
        }
        """
    )

st.markdown("---")
st.subheader("3) IDS Prediction Panel")
feature_names = [f"f{i}" for i in range(x_test.shape[1])]
orig_sample = x_test[row_index]
orig_pred, orig_conf = batch_predict_and_confidence(model, orig_sample.reshape(1, -1))

p1, p2 = st.columns([2.2, 1.2])
with p1:
    st.dataframe(pd.DataFrame([orig_sample], columns=feature_names), use_container_width=True)
with p2:
    card("Predicted Label", label_name(orig_pred[0]))
    card("Confidence", f"{orig_conf[0] * 100:.2f}%")

st.markdown("---")
st.subheader("4) Attack Simulation Panel")
attack_clicked = st.button("Apply UAP Attack", type="primary")

if attack_clicked:
    result = compute_bundle(model, x_test, y_test, uap, constraints, epsilon, eval_size)
    st.session_state["bundle"] = result
    st.session_state["epsilon"] = epsilon

if "bundle" not in st.session_state:
    st.info("Click Apply UAP Attack to run perturbation, re-predict, and update all panels.")
else:
    bundle = st.session_state["bundle"]
    eps_used = st.session_state["epsilon"]

    adv_sample = apply_uap_batch(orig_sample.reshape(1, -1), uap, constraints, eps_used)[0]
    adv_pred, adv_conf = batch_predict_and_confidence(model, adv_sample.reshape(1, -1))

    diff = adv_sample - orig_sample
    changed = np.abs(diff) > 1e-9

    compare_df = pd.DataFrame(
        {
            "feature": feature_names,
            "original": orig_sample,
            "modified": adv_sample,
            "delta": diff,
            "changed": changed,
        }
    )

    def style_changes(row):
        return ["background-color: #ffe7a3" if row["changed"] else "" for _ in row]

    st.write(f"Applied epsilon: {eps_used:.3f}")
    st.dataframe(compare_df.style.apply(style_changes, axis=1), use_container_width=True)

    st.markdown("---")
    st.subheader("5) Comparison Panel")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Original Prediction", f"{label_name(orig_pred[0])} ({orig_conf[0] * 100:.1f}%)")
    with c2:
        card("Modified Prediction", f"{label_name(adv_pred[0])} ({adv_conf[0] * 100:.1f}%)")
    with c3:
        changed_flag = "CHANGED" if int(orig_pred[0]) != int(adv_pred[0]) else "UNCHANGED"
        color = "#ff4d4f" if changed_flag == "CHANGED" else "#2d7"
        st.markdown(
            f"<div style='padding:14px;border:2px solid {color};border-radius:12px;'><b>Decision Shift:</b> {changed_flag}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("6) Metrics Dashboard")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        card("Accuracy (Before)", f"{bundle['acc_before'] * 100:.2f}%")
    with m2:
        card("Accuracy (After)", f"{bundle['acc_after'] * 100:.2f}%")
    with m3:
        card("FNR (Before)", f"{bundle['fnr_before'] * 100:.2f}%")
    with m4:
        card("FNR (After)", f"{bundle['fnr_after'] * 100:.2f}%")
    with m5:
        card("Fooling Rate", f"{bundle['fooling_rate'] * 100:.2f}%")

    st.markdown("---")
    st.subheader("7) Graph Section")
    curve_df = build_curves(model, x_test, y_test, uap, constraints, eval_size)

    g1, g2, g3 = st.columns(3)
    with g1:
        fig_acc = px.line(curve_df, x="epsilon", y="accuracy", markers=True, title="Accuracy vs Epsilon")
        st.plotly_chart(fig_acc, use_container_width=True)
    with g2:
        fig_fnr = px.line(curve_df, x="epsilon", y="fnr", markers=True, title="FNR vs Epsilon")
        st.plotly_chart(fig_fnr, use_container_width=True)
    with g3:
        fig_fool = px.line(curve_df, x="epsilon", y="fooling_rate", markers=True, title="Fooling Rate vs Epsilon")
        st.plotly_chart(fig_fool, use_container_width=True)

    st.markdown("---")
    st.subheader("8) Confusion Matrix")
    cm1, cm2 = st.columns(2)

    with cm1:
        cm_before = bundle["cm_before"]
        fig_cm_before = px.imshow(
            cm_before,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Pred Benign", "Pred Attack"],
            y=["True Benign", "True Attack"],
            title="Before Attack",
        )
        st.plotly_chart(fig_cm_before, use_container_width=True)

    with cm2:
        cm_after = bundle["cm_after"]
        fig_cm_after = px.imshow(
            cm_after,
            text_auto=True,
            color_continuous_scale="Reds",
            x=["Pred Benign", "Pred Attack"],
            y=["True Benign", "True Attack"],
            title="After Attack",
        )
        st.plotly_chart(fig_cm_after, use_container_width=True)

    st.markdown("---")
    st.subheader("9) Feature Impact Visualization")
    base_eps = float(np.max(np.abs(uap)))
    scale = (eps_used / base_eps) if base_eps > 1e-12 else 0.0
    uap_scaled = uap * scale
    impact = np.abs(uap_scaled)

    top_k = min(20, len(impact))
    top_idx = np.argsort(-impact)[:top_k]
    impact_df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in top_idx],
            "abs_delta": impact[top_idx],
        }
    )

    fig_imp = px.bar(
        impact_df,
        x="feature",
        y="abs_delta",
        title="Most Influential Modified Features",
        color="abs_delta",
        color_continuous_scale="Oranges",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.subheader("System Behavior")
st.write(
    "When Apply UAP Attack is clicked, the dashboard perturbs input data, recalculates dependent features, re-runs IDS prediction, and refreshes metrics, curves, confusion matrices, and feature impact charts."
)

try:
    with open("trained_models/forward_metrics.json", "r", encoding="utf-8") as f:
        cached_metrics = json.load(f)
    with st.expander("Last saved forward run metrics"):
        st.json(cached_metrics)
except FileNotFoundError:
    pass
