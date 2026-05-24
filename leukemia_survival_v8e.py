# %% [markdown]
# # V8e OOF Rank Blend Diagnostics
#
# This script uses saved V8cd OOF vectors only. It does not retrain models and
# does not generate submission CSVs.

# %%
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv


TAU = 7.0
V6_BASELINE = 0.7140458961158811
V7B_BASELINE = 0.7152546075843262
ACCEPTANCE_THRESHOLD = 0.7202546075843262

DATA_DIR = Path(r"C:\Users\maze2\Downloads\QRT_blood")
OUTPUT_DIR = Path.cwd() / "qrt_outputs_v8"
OOF_PATH = OUTPUT_DIR / "v8cd_oof_predictions.npz"


# %% [markdown]
# # Load Outcomes And OOF Vectors

# %%
target = pd.read_csv(DATA_DIR / "target_train.csv")
target_clean = target.dropna(subset=["OS_YEARS", "OS_STATUS"]).copy()
y = Surv.from_arrays(
    event=target_clean["OS_STATUS"].astype(bool).to_numpy(),
    time=target_clean["OS_YEARS"].astype(float).to_numpy(),
)

if not OOF_PATH.exists():
    raise FileNotFoundError(f"Missing saved OOF artifact: {OOF_PATH}")

with np.load(OOF_PATH) as loaded:
    saved_oof = {key: loaded[key] for key in loaded.files}

MODEL_NAMES = [
    "rsf",
    "extra",
    "gbsa",
    "coxnet",
    "coxph",
    "xgb_cox_d2_lr03",
    "xgb_cox_d3_lr02",
    "xgb_aft_d2_lr03_normal",
    "xgb_aft_d3_lr02_normal",
    "lgbm_horizon_l15_lr03",
    "lgbm_horizon_l31_lr02",
]


def rank01(values):
    values = np.asarray(values, dtype=float)
    return pd.Series(values).rank(method="average").to_numpy(dtype=float) / (len(values) + 1.0)


def score_ipcw(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(y),):
        raise ValueError(f"Bad OOF shape {values.shape}; expected {(len(y),)}")
    if not np.isfinite(values).all():
        raise ValueError("OOF values contain non-finite values")
    return float(concordance_index_ipcw(y, y, values, tau=TAU)[0])


def get_model_oof(name):
    if name in saved_oof:
        return saved_oof[name]
    pred_key = f"oof_pred__{name}"
    if pred_key in saved_oof:
        return saved_oof[pred_key]
    raise KeyError(f"Missing OOF vector for {name}")


oof_raw = {name: get_model_oof(name) for name in MODEL_NAMES}
oof_checks = []
for name, values in oof_raw.items():
    oof_checks.append(
        {
            "model": name,
            "shape": str(values.shape),
            "finite": bool(np.isfinite(values).all()),
            "available": True,
        }
    )
oof_check_df = pd.DataFrame(oof_checks)
print(oof_check_df.to_string(index=False))
assert oof_check_df["finite"].all()

oof_ranks = {name: rank01(values) for name, values in oof_raw.items()}


# %% [markdown]
# # Individual Scores And Correlations

# %%
individual_scores = pd.DataFrame(
    [
        {
            "model": name,
            "oof_ipcw": score_ipcw(oof_ranks[name]),
            "delta_vs_v6": score_ipcw(oof_ranks[name]) - V6_BASELINE,
            "delta_vs_v7b": score_ipcw(oof_ranks[name]) - V7B_BASELINE,
            "clears_threshold": score_ipcw(oof_ranks[name]) >= ACCEPTANCE_THRESHOLD,
        }
        for name in MODEL_NAMES
    ]
).sort_values("oof_ipcw", ascending=False)

rank_corr = pd.DataFrame(oof_ranks).corr(method="spearman")

print("\nIndividual model OOF IPCW:")
print(individual_scores.to_string(index=False))
print("\nSpearman rank correlation matrix:")
print(rank_corr.round(4).to_string())

individual_scores.to_csv(OUTPUT_DIR / "v8e_individual_oof_ipcw.csv", index=False)
rank_corr.to_csv(OUTPUT_DIR / "v8e_rank_correlation_matrix.csv")
oof_check_df.to_csv(OUTPUT_DIR / "v8e_oof_checks.csv", index=False)


# %% [markdown]
# # Rank Blend Candidates

# %%
def blend_score(name, weights):
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Blend has non-positive total weight")
    blended = np.zeros(len(y), dtype=float)
    normalized = {}
    for model_name, weight in weights.items():
        normalized[model_name] = float(weight) / total
        blended += normalized[model_name] * oof_ranks[model_name]
    blended_rank = rank01(blended)
    blend_ipcw = score_ipcw(blended_rank)
    return {
        "blend": name,
        "oof_ipcw": blend_ipcw,
        "delta_vs_v6": blend_ipcw - V6_BASELINE,
        "delta_vs_v7b": blend_ipcw - V7B_BASELINE,
        "clears_threshold": blend_ipcw >= ACCEPTANCE_THRESHOLD,
        "weights": normalized,
        "models": list(normalized.keys()),
    }


blend_rows = [
    blend_score("rsf alone", {"rsf": 1.0}),
    blend_score("extra alone", {"extra": 1.0}),
    blend_score("rsf + extra", {"rsf": 1.0, "extra": 1.0}),
    blend_score("rsf + extra + gbsa", {"rsf": 1.0, "extra": 1.0, "gbsa": 1.0}),
    blend_score(
        "rsf + extra + xgb_cox_d3_lr02",
        {"rsf": 1.0, "extra": 1.0, "xgb_cox_d3_lr02": 1.0},
    ),
    blend_score(
        "rsf + extra + gbsa + xgb_cox_d3_lr02",
        {"rsf": 1.0, "extra": 1.0, "gbsa": 1.0, "xgb_cox_d3_lr02": 1.0},
    ),
]


# %% [markdown]
# # Greedy Forward Selection

# %%
remaining = set(MODEL_NAMES)
selected = []
current_score = -np.inf

while remaining:
    trials = []
    for candidate in sorted(remaining):
        weights = {name: 1.0 for name in selected + [candidate]}
        row = blend_score("trial", weights)
        trials.append((candidate, row["oof_ipcw"], row))
    best_candidate, best_score, best_row = max(trials, key=lambda item: item[1])
    if best_score > current_score:
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_score = best_score
    else:
        break

greedy_weights = {name: 1.0 for name in selected}
blend_rows.append(blend_score(f"greedy forward ({selected})", greedy_weights))


# %% [markdown]
# # Small Grid Over Four Main Models

# %%
grid_models = ["rsf", "extra", "gbsa", "xgb_cox_d3_lr02"]
grid_values = [0, 1, 2, 3, 4]
grid_rows = []
for combo in itertools.product(grid_values, repeat=len(grid_models)):
    if sum(combo) == 0:
        continue
    weights = {model: weight for model, weight in zip(grid_models, combo) if weight > 0}
    row = blend_score("grid", weights)
    row["integer_weights"] = dict(zip(grid_models, combo))
    grid_rows.append(row)

grid_df = pd.DataFrame(grid_rows).sort_values("oof_ipcw", ascending=False)
best_grid = grid_df.iloc[0].to_dict()
blend_rows.append(
    {
        "blend": "best grid rsf/extra/gbsa/xgb_cox_d3",
        "oof_ipcw": best_grid["oof_ipcw"],
        "delta_vs_v6": best_grid["delta_vs_v6"],
        "delta_vs_v7b": best_grid["delta_vs_v7b"],
        "clears_threshold": best_grid["clears_threshold"],
        "weights": best_grid["weights"],
        "models": best_grid["models"],
    }
)


# %% [markdown]
# # Save Diagnostics

# %%
blend_df = pd.DataFrame(blend_rows).sort_values("oof_ipcw", ascending=False)
best_blend = blend_df.iloc[0]

print("\nBlend scores:")
print(blend_df.to_string(index=False))
print("\nBest blend:")
print(best_blend.to_string())

blend_df.to_csv(OUTPUT_DIR / "v8e_blend_scores.csv", index=False)
grid_df.to_csv(OUTPUT_DIR / "v8e_grid_weight_scores.csv", index=False)
blend_df.to_csv(OUTPUT_DIR / "v8e_no_submission_summary.csv", index=False)

print("\nNo submission CSVs generated.")
