# %% [markdown]
# # Step 1: Colab setup
#
# This notebook builds V8a leakage-safe survival target-alignment diagnostics
# for the leukemia challenge. V8a compares full survival targets with tau=7
# administratively censored targets using the V6 feature set only.
#
# Long-running cells:
# - Installing `scikit-survival` can take 2-5 minutes.
# - Random KFold diagnostics can take 10-25 minutes.
# - GroupKFold model selection can take 20-45 minutes.
# - LOCO diagnostics can take 10-30+ minutes depending on toggles.

# %%
# Colab only: install survival-analysis dependencies.
try:
    import sksurv  # noqa: F401
except Exception:
    try:
        import google.colab  # noqa: F401
        get_ipython().system("pip -q install scikit-survival")
    except Exception as exc:
        raise ImportError(
            "scikit-survival is required. In Colab, run: !pip -q install scikit-survival"
        ) from exc

import os
import zipfile
import warnings
import re
import gc
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

try:
    from sksurv.ensemble import ExtraSurvivalTrees
    HAS_EXTRA_SURVIVAL_TREES = True
except Exception as exc:
    print(f"ExtraSurvivalTrees unavailable; skipping it. Reason: {exc}")
    ExtraSurvivalTrees = None
    HAS_EXTRA_SURVIVAL_TREES = False

RANDOM_STATE = 42
TAU = 7.0

def env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


RUN_RANDOM_KFOLD_DIAGNOSTIC = env_flag("RUN_RANDOM_KFOLD_DIAGNOSTIC", False)
RUN_LOCO_DIAGNOSTIC = env_flag("RUN_LOCO_DIAGNOSTIC", False)
ENABLE_EXTRA_SURVIVAL_TREES = env_flag("ENABLE_EXTRA_SURVIVAL_TREES", True)
FEATURE_COUNTS_ONLY = env_flag("FEATURE_COUNTS_ONLY", False)
SAVE_V8_SUBMISSIONS = env_flag("SAVE_V8_SUBMISSIONS", False)

TARGET_ORDER = ["full", "tau7"]

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
np.random.seed(RANDOM_STATE)

try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)


# %% [markdown]
# # Step 2: File paths
#
# In Colab, put the five challenge files in `/content/QRT_blood`.
# The code also auto-detects the local Windows download path used in this project.

# %%
def first_existing_dir(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_DIR = first_existing_dir([
    Path("/content/QRT_blood"),
    Path("/content"),
    Path(r"C:\Users\maze2\Downloads\QRT_blood"),
])

OUTPUT_DIR = first_existing_dir([
    Path("/content"),
    Path.cwd(),
]) / "qrt_outputs_v8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ZIP = DATA_DIR / "X_train_9po2I7U.zip"
TEST_ZIP = DATA_DIR / "X_test_xzVefmA.zip"
TARGET_PATH = DATA_DIR / "target_train.csv"
RANDOM_SUB_PATH = DATA_DIR / "random_submission_FRacdcw_v9kP4pP.csv"

# Prior submissions are diagnostics only. They are never merged into features,
# training, stacking, ensembling, calibration, or final risk generation.
PRIOR_SUBMISSION_PATHS = [
    DATA_DIR.parent / "submission_v3_cox_nmut.csv",
    DATA_DIR.parent / "submission_v4_best_features.csv",
    DATA_DIR.parent / "submission_v5_best_single_rsf_deeper.csv",
    DATA_DIR.parent / "submission_v5b_top3_weighted_ensemble.csv",
    Path("/content/submission_v3_cox_nmut.csv"),
    Path("/content/submission_v4_best_features.csv"),
    Path("/content/submission_v5_best_single_rsf_deeper.csv"),
    Path("/content/submission_v5b_top3_weighted_ensemble.csv"),
]
PRIOR_SUBMISSION_PATHS = [p for p in PRIOR_SUBMISSION_PATHS if p.exists()]

print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
for required in [TRAIN_ZIP, TEST_ZIP, TARGET_PATH, RANDOM_SUB_PATH]:
    print(required.name, "exists:", required.exists())


# %% [markdown]
# # Step 3: Load raw files

# %%
def read_csv_from_zip(zip_path, inner_path):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path) as fh:
            return pd.read_csv(fh)


clinical_train = read_csv_from_zip(TRAIN_ZIP, "X_train/clinical_train.csv")
molecular_train = read_csv_from_zip(TRAIN_ZIP, "X_train/molecular_train.csv")
clinical_test = read_csv_from_zip(TEST_ZIP, "X_test/clinical_test.csv")
molecular_test = read_csv_from_zip(TEST_ZIP, "X_test/molecular_test.csv")
target = pd.read_csv(TARGET_PATH)
random_submission = pd.read_csv(RANDOM_SUB_PATH)

prior_submissions = {}
for path in PRIOR_SUBMISSION_PATHS:
    try:
        prior_submissions[path.stem] = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read prior submission {path}: {exc}")

print("clinical_train:", clinical_train.shape)
print("molecular_train:", molecular_train.shape)
print("target:", target.shape)
print("clinical_test:", clinical_test.shape)
print("molecular_test:", molecular_test.shape)
print("random_submission:", random_submission.shape)


# %% [markdown]
# # Step 4: Data structure and ID checks
#
# These assertions protect against row-order mistakes and accidental leakage.

# %%
assert set(clinical_train["ID"]) == set(target["ID"]), "Train clinical IDs and target IDs differ."
assert list(clinical_train["ID"]) == list(target["ID"]), "Train clinical and target row order differ."
assert set(clinical_test["ID"]) == set(random_submission["ID"]), "Test clinical IDs and random submission IDs differ."
assert list(clinical_test["ID"]) == list(random_submission["ID"]), "Test clinical and random submission row order differ."
assert set(molecular_train["ID"]).issubset(set(clinical_train["ID"])), "Train molecular has IDs not in clinical."
assert set(molecular_test["ID"]).issubset(set(clinical_test["ID"])), "Test molecular has IDs not in clinical."

print("Clinical columns:", list(clinical_train.columns))
print("Molecular columns:", list(molecular_train.columns))
print("Target columns:", list(target.columns))
print("Missing outcomes:", target[["OS_YEARS", "OS_STATUS"]].isna().any(axis=1).sum())


# %% [markdown]
# # Step 5: Diagnostics only: train/test shift and prior submissions
#
# This cell prints useful information but does not create model-selection inputs.
# `CENTER` is diagnostics/splitting only. Prior submissions are correlations only.

# %%
NUMERIC_CLINICAL_COLS = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]

print("\nTrain centers:")
print(clinical_train["CENTER"].value_counts(dropna=False))
print("\nTest centers:")
print(clinical_test["CENTER"].value_counts(dropna=False))
print("\nCenters only in test:", sorted(set(clinical_test["CENTER"].dropna()) - set(clinical_train["CENTER"].dropna())))

missing_report = pd.DataFrame({
    "train_missing_pct": clinical_train[NUMERIC_CLINICAL_COLS + ["CYTOGENETICS"]].isna().mean() * 100,
    "test_missing_pct": clinical_test[NUMERIC_CLINICAL_COLS + ["CYTOGENETICS"]].isna().mean() * 100,
}).round(1)
print("\nMissingness:")
display(missing_report)

median_report = pd.DataFrame({
    "train_median": clinical_train[NUMERIC_CLINICAL_COLS].median(numeric_only=True),
    "test_median": clinical_test[NUMERIC_CLINICAL_COLS].median(numeric_only=True),
}).round(3)
print("\nClinical medians:")
display(median_report)

for label, mol, clin in [
    ("train", molecular_train, clinical_train),
    ("test", molecular_test, clinical_test),
]:
    mut_counts = mol.groupby("ID").size().reindex(clin["ID"], fill_value=0)
    print(
        f"\n{label} mutation burden:",
        {
            "rows": len(mol),
            "patients_with_mut": int((mut_counts > 0).sum()),
            "mean": float(mut_counts.mean()),
            "median": float(mut_counts.median()),
            "p95": float(mut_counts.quantile(0.95)),
            "max": int(mut_counts.max()),
        },
    )
    print(f"{label} top genes:", mol["GENE"].value_counts().head(15).to_dict())
    print(f"{label} top effects:", mol["EFFECT"].value_counts().head(10).to_dict())

if prior_submissions:
    aligned_prior = pd.DataFrame({"ID": random_submission["ID"]})
    for name, df_prior in prior_submissions.items():
        if set(aligned_prior["ID"]).issubset(set(df_prior["ID"])) and "risk_score" in df_prior.columns:
            aligned_prior[name] = df_prior.set_index("ID").loc[aligned_prior["ID"], "risk_score"].values
    if aligned_prior.shape[1] > 1:
        print("\nPrior submission Spearman correlations, diagnostics only:")
        display(aligned_prior.drop(columns="ID").corr(method="spearman").round(4))


# %% [markdown]
# # Step 6: Clean supervised target
#
# Rows with missing outcomes are excluded from supervised fitting and CV.

# %%
target_clean = target.dropna(subset=["OS_YEARS", "OS_STATUS"]).copy()
target_clean["event_full"] = target_clean["OS_STATUS"].astype(bool)
target_clean["time_full"] = target_clean["OS_YEARS"].astype(float)

train_ids = target_clean["ID"].tolist()
test_ids = random_submission["ID"].tolist()

groups_center = clinical_train.set_index("ID").loc[train_ids, "CENTER"].values
y_full = Surv.from_arrays(
    event=target_clean["event_full"].to_numpy(dtype=bool),
    time=target_clean["time_full"].to_numpy(dtype=float),
)

event_tau7 = (
    target_clean["event_full"].to_numpy(dtype=bool)
    & (target_clean["time_full"].to_numpy(dtype=float) <= TAU)
)
time_tau7 = np.minimum(target_clean["time_full"].to_numpy(dtype=float), TAU)

y_tau7 = Surv.from_arrays(
    event=event_tau7,
    time=time_tau7,
)

TARGET_CONFIGS = {
    "full": y_full,
    "tau7": y_tau7,
}

# Existing utility functions use `y` for scoring/splitting; keep it bound to
# the original target because IPCW evaluation is always against full outcomes.
y = y_full

print("Clean training rows:", len(train_ids))
print("Full target event rate:", round(float(target_clean["event_full"].mean()), 4))
print("Tau7 target event rate:", round(float(event_tau7.mean()), 4))
print("Max full time:", target_clean["time_full"].max())
print("Max tau7 time:", float(time_tau7.max()))


# %% [markdown]
# # Step 7: Clinical features
#
# Build deterministic clinical features with NaNs preserved. No imputation here.
# `CENTER` is deliberately excluded.

# %%
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_divide(num, den):
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def make_clinical_features(df):
    f = pd.DataFrame(index=df["ID"].astype(str))
    raw = {}
    for col in NUMERIC_CLINICAL_COLS:
        raw[col] = safe_numeric(df[col])
        f[col] = raw[col].to_numpy()
        f[f"{col}_missing"] = raw[col].isna().astype(float).to_numpy()
        f[f"log1p_{col}"] = np.log1p(raw[col].clip(lower=0)).to_numpy()

    f["ANC_WBC_ratio"] = safe_divide(raw["ANC"], raw["WBC"]).to_numpy()
    f["MONOCYTES_WBC_ratio"] = safe_divide(raw["MONOCYTES"], raw["WBC"]).to_numpy()
    f["PLT_HB_ratio"] = safe_divide(raw["PLT"], raw["HB"]).to_numpy()
    f["WBC_PLT_ratio"] = safe_divide(raw["WBC"], raw["PLT"]).to_numpy()
    f["blast_x_wbc"] = (raw["BM_BLAST"] * raw["WBC"]).to_numpy()
    f["blast_x_logwbc"] = (raw["BM_BLAST"] * np.log1p(raw["WBC"].clip(lower=0))).to_numpy()
    f["anemia_thrombocytopenia"] = ((raw["HB"] < 10) & (raw["PLT"] < 100)).astype(float).to_numpy()
    f["severe_cytopenia_count"] = (
        (raw["HB"] < 8).astype(float)
        + (raw["PLT"] < 50).astype(float)
        + (raw["ANC"] < 0.5).astype(float)
    ).to_numpy()

    thresholds = {
        "BM_BLAST": [2, 5, 10, 20, 30],
        "WBC": [1, 3, 10, 20, 50],
        "ANC": [0.5, 1.0, 1.5],
        "MONOCYTES": [0.2, 0.5, 1.0],
        "HB": [8, 10, 12],
        "PLT": [20, 50, 100, 150],
    }
    for col, vals in thresholds.items():
        for val in vals:
            op = "lt" if col in ["ANC", "HB", "PLT"] else "ge"
            if op == "lt":
                f[f"{col}_{op}_{str(val).replace('.', 'p')}"] = (raw[col] < val).astype(float).to_numpy()
            else:
                f[f"{col}_{op}_{str(val).replace('.', 'p')}"] = (raw[col] >= val).astype(float).to_numpy()

    assert "CENTER" not in f.columns
    return f


def make_v7e_clinical_features(df):
    """Small fixed domain-threshold add-ons; NaNs are preserved as false flags."""
    f = pd.DataFrame(index=df["ID"].astype(str))
    bm = safe_numeric(df["BM_BLAST"])
    wbc = safe_numeric(df["WBC"])
    anc = safe_numeric(df["ANC"])
    mono = safe_numeric(df["MONOCYTES"])
    hb = safe_numeric(df["HB"])
    plt = safe_numeric(df["PLT"])
    monocyte_fraction = safe_divide(mono, wbc)

    f["clin__bm_blast_ge_10"] = (bm >= 10).astype(float).to_numpy()
    f["clin__bm_blast_ge_20"] = (bm >= 20).astype(float).to_numpy()
    f["clin__hyperleukocytosis_wbc_gt_100"] = (wbc > 100).astype(float).to_numpy()
    f["clin__wbc_gt_25"] = (wbc > 25).astype(float).to_numpy()
    f["clin__anc_lt_1_8"] = (anc < 1.8).astype(float).to_numpy()
    f["clin__plt_lt_150"] = (plt < 150).astype(float).to_numpy()
    f["clin__hb_lt_10"] = (hb < 10).astype(float).to_numpy()
    f["count__cytopenia_count_simple"] = (
        f["clin__anc_lt_1_8"] + f["clin__plt_lt_150"] + f["clin__hb_lt_10"]
    )
    f["clin__absolute_monocytosis_ge_0_5"] = (mono >= 0.5).astype(float).to_numpy()
    f["clin__monocyte_fraction_ge_0_10"] = (monocyte_fraction >= 0.10).astype(float).to_numpy()
    f["int__monocytosis_proxy_both"] = (
        (f["clin__absolute_monocytosis_ge_0_5"] == 1)
        & (f["clin__monocyte_fraction_ge_0_10"] == 1)
    ).astype(float)
    return f


clinical_features_train = make_clinical_features(clinical_train)
clinical_features_test = make_clinical_features(clinical_test)
print("Clinical feature count:", clinical_features_train.shape[1])


# %% [markdown]
# # Step 8: Cytogenetic features
#
# Fixed karyotype parser. No data-derived selection from test.

# %%
CYTO_PATTERNS = {
    "cyto_del5q": r"del\(5\)|5q-|del\(5q\)",
    "cyto_monosomy5": r"(?<!\d)-5(?!\d)",
    "cyto_del7q": r"del\(7\)|7q-|del\(7q\)",
    "cyto_monosomy7": r"(?<!\d)-7(?!\d)",
    "cyto_plus8": r"(?<!\d)\+8(?!\d)",
    "cyto_del11q": r"del\(11\)|11q-|del\(11q\)",
    "cyto_del12p": r"del\(12\)|12p-|del\(12p\)",
    "cyto_del13q": r"del\(13\)|13q-|del\(13q\)",
    "cyto_del17p": r"del\(17\)|17p-|del\(17p\)|i\(17q\)",
    "cyto_del20q": r"del\(20\)|20q-|del\(20q\)",
    "cyto_inv3_or_t3q": r"inv\(3\)|t\(3;3\)|3q26|q26\.2",
    "cyto_t_8_21": r"t\(8;21\)",
    "cyto_inv16": r"inv\(16\)|t\(16;16\)",
    "cyto_t_15_17": r"t\(15;17\)",
    "cyto_kmt2a_11q23": r"11q23|q23",
    "cyto_marker_or_ring": r"mar|r\(",
    "cyto_derivative": r"der\(",
    "cyto_dicentric": r"dic\(",
    "cyto_additional": r"add\(",
    "cyto_translocation": r"t\(",
    "cyto_inversion": r"inv\(",
}


def normalize_cyto_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(" ", "")


def make_cytogenetic_features(df):
    texts = df["CYTOGENETICS"].map(normalize_cyto_text)
    f = pd.DataFrame(index=df["ID"].astype(str))
    f["cyto_missing"] = (texts == "").astype(float).to_numpy()
    f["cyto_text_len"] = texts.str.len().astype(float).to_numpy()
    f["cyto_clone_count"] = texts.str.count("/").add(1).where(texts != "", 0).astype(float).to_numpy()
    f["cyto_uncertain_count"] = texts.str.count(r"\?").astype(float).to_numpy()
    f["cyto_has_uncertainty"] = (f["cyto_uncertain_count"] > 0).astype(float)
    f["cyto_normal_word"] = texts.str.contains("normal", regex=False).astype(float).to_numpy()
    f["cyto_plain_46xx_or_46xy"] = texts.isin(["46,xx", "46,xy"]).astype(float).to_numpy()

    event_regex = r"del\(|add\(|der\(|dic\(|inv\(|ins\(|t\(|mar|r\(|(?<!\d)[+-](?:[1-9]|1[0-9]|2[0-2]|x|y)(?!\d)"
    f["cyto_event_count_est"] = texts.str.count(event_regex).astype(float).to_numpy()
    f["cyto_abnormal"] = ((f["cyto_event_count_est"] > 0) | (f["cyto_normal_word"] == 0)).astype(float)

    for name, pattern in CYTO_PATTERNS.items():
        f[name] = texts.str.contains(pattern, regex=True).astype(float).to_numpy()

    f["cyto_minus7_or_del7q"] = ((f["cyto_monosomy7"] > 0) | (f["cyto_del7q"] > 0)).astype(float)
    f["cyto_minus5_or_del5q"] = ((f["cyto_monosomy5"] > 0) | (f["cyto_del5q"] > 0)).astype(float)
    f["cyto_complex_3plus"] = (f["cyto_event_count_est"] >= 3).astype(float)
    f["cyto_complex_5plus"] = (f["cyto_event_count_est"] >= 5).astype(float)

    adverse_cols = [
        "cyto_minus7_or_del7q",
        "cyto_minus5_or_del5q",
        "cyto_del17p",
        "cyto_inv3_or_t3q",
        "cyto_kmt2a_11q23",
        "cyto_complex_3plus",
    ]
    favorable_cols = ["cyto_t_8_21", "cyto_inv16", "cyto_t_15_17"]
    f["cyto_adverse_score"] = f[adverse_cols].sum(axis=1)
    f["cyto_favorable_score"] = f[favorable_cols].sum(axis=1)
    f["cyto_intermediate_like"] = ((f["cyto_adverse_score"] == 0) & (f["cyto_favorable_score"] == 0)).astype(float)

    for chrom in [str(i) for i in range(1, 23)] + ["x", "y"]:
        f[f"cyto_chr_{chrom}_mentioned"] = texts.str.contains(
            rf"(?<!\d){chrom}(?!\d)|\({chrom}[;,\)]|;{chrom}[;,\)]", regex=True
        ).astype(float).to_numpy()

    return f


UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"


def norm_cyto(value):
    if pd.isna(value):
        return ""
    s = str(value).upper().strip()
    for dash in UNICODE_DASHES:
        s = s.replace(dash, "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([,;/()\\[\\]])\s*", r"\1", s)
    s = s.replace("AML1-ETO", "RUNX1::RUNX1T1")
    s = s.replace("PML-RARA", "PML::RARA")
    s = s.replace("CBFB-MYH11", "CBFB::MYH11")
    s = s.replace("BCR-ABL1", "BCR::ABL1")
    s = re.sub(r"\bMLL\b", "KMT2A", s)
    return s


def has_pattern(texts, pattern):
    return texts.str.contains(pattern, regex=True, na=False).astype(float)


def strip_clone_counts(text):
    return re.sub(r"\[[^\]]*\]", "", text)


STRUCTURAL_TOKEN_RE = re.compile(
    r"(?:T|INV|DEL|ADD|DUP|INS|DER|DIC|IDIC|I|R)\([^)]*\)(?:\([^)]*\))*|(?:^|[,;/])MAR\d*",
    re.IGNORECASE,
)
AUTOSOMAL_GAIN_LOSS_RE = re.compile(r"(?<![A-Z0-9])([+-])([1-9]|1[0-9]|2[0-2])(?![0-9])")


def parse_iscn_abnormalities(text):
    s = strip_clone_counts(norm_cyto(text))
    if not s:
        return {
            "abn_count": 0,
            "structural_abn_count": 0,
            "autosomal_monosomy_count": 0,
            "gain_count": 0,
            "pure_multi_trisomy_no_structure": False,
        }

    tokens = set()
    structural = set()
    autosomal_monosomies = set()
    autosomal_gains = set()

    for clone in s.split("/"):
        clone = re.sub(r"^\d{2,3},(?:XX|XY|X|-Y|\+Y|XXY|XYY|XXX),?", "", clone)
        for match in STRUCTURAL_TOKEN_RE.finditer(clone):
            token = match.group(0).strip(",;/")
            structural.add(token)
            tokens.add(token)
        for sign, chrom in AUTOSOMAL_GAIN_LOSS_RE.findall(clone):
            token = f"{sign}{chrom}"
            tokens.add(token)
            if sign == "-":
                autosomal_monosomies.add(chrom)
            elif sign == "+":
                autosomal_gains.add(chrom)

    pure_multi_trisomy_no_structure = (
        len(autosomal_gains) >= 3
        and len(structural) == 0
        and len(autosomal_monosomies) == 0
    )
    return {
        "abn_count": len(tokens),
        "structural_abn_count": len(structural),
        "autosomal_monosomy_count": len(autosomal_monosomies),
        "gain_count": len(autosomal_gains),
        "pure_multi_trisomy_no_structure": pure_multi_trisomy_no_structure,
    }


def make_v7a_cytogenetic_features(df):
    texts = df["CYTOGENETICS"].map(norm_cyto)
    f = pd.DataFrame(index=df["ID"].astype(str))

    f["cyto__t_8_21_runx1_runx1t1"] = has_pattern(
        texts, r"T\(8;21\)|RUNX1::RUNX1T1|RUNX1-RUNX1T1|AML1-ETO"
    ).to_numpy()
    f["cyto__inv_16_or_t_16_16_cbfb_myh11"] = has_pattern(
        texts, r"INV\(16\)|T\(16;16\)|CBFB::MYH11|CBFB-MYH11"
    ).to_numpy()
    f["cyto__cbf_any"] = (
        (f["cyto__t_8_21_runx1_runx1t1"] == 1)
        | (f["cyto__inv_16_or_t_16_16_cbfb_myh11"] == 1)
    ).astype(float)
    f["cyto__apl_like_t_15_17_pml_rara"] = has_pattern(
        texts, r"T\(15;17\)|PML::RARA|PML-RARA"
    ).to_numpy()
    f["cyto__t_9_11_mllt3_kmt2a"] = has_pattern(
        texts, r"T\(9;11\)|MLLT3::KMT2A|KMT2A::MLLT3"
    ).to_numpy()
    kmt2a_any = has_pattern(texts, r"11Q23|KMT2A|MLL").astype(bool)
    kmt2a_ptd = has_pattern(texts, r"PTD").astype(bool)
    f["cyto__kmt2a_rearranged_other"] = (
        kmt2a_any.to_numpy()
        & (f["cyto__t_9_11_mllt3_kmt2a"].to_numpy() == 0)
        & (~kmt2a_ptd.to_numpy())
    ).astype(float)
    f["cyto__t_6_9_dek_nup214"] = has_pattern(
        texts, r"T\(6;9\)|DEK::NUP214|DEK-NUP214"
    ).to_numpy()
    f["cyto__t_9_22_bcr_abl1_like"] = has_pattern(
        texts, r"T\(9;22\)|BCR::ABL1|BCR-ABL1"
    ).to_numpy()
    f["cyto__inv_3_or_t_3_3_or_3q26_meccom"] = has_pattern(
        texts, r"INV\(3\)|T\(3;3\)|3Q26|MECOM|EVI1"
    ).to_numpy()
    f["cyto__chr5_abn"] = has_pattern(
        texts, r"(?<![A-Z0-9])-5(?![0-9])|DEL\(5Q\)|DEL\(5\)\(Q[^)]*\)"
    ).to_numpy()
    f["cyto__chr7_abn"] = has_pattern(
        texts, r"(?<![A-Z0-9])-7(?![0-9])|DEL\(7Q\)|DEL\(7\)\(Q[^)]*\)"
    ).to_numpy()
    f["cyto__chr17p_abn"] = has_pattern(
        texts, r"(?<![A-Z0-9])-17(?![0-9])|ABN\(17P\)|DEL\(17P\)|DEL\(17\)\(P[^)]*\)"
    ).to_numpy()
    f["cyto__normal_karyotype_strict"] = texts.str.match(
        r"^46,(XX|XY)(\[\d+\])?$", na=False
    ).astype(float).to_numpy()

    parsed = texts.map(parse_iscn_abnormalities)
    abn_count = parsed.map(lambda d: d["abn_count"]).astype(float)
    structural_count = parsed.map(lambda d: d["structural_abn_count"]).astype(float)
    monosomy_count = parsed.map(lambda d: d["autosomal_monosomy_count"]).astype(float)
    pure_multi_trisomy = parsed.map(lambda d: d["pure_multi_trisomy_no_structure"]).astype(bool)

    recurrent_cols = [
        "cyto__t_8_21_runx1_runx1t1",
        "cyto__inv_16_or_t_16_16_cbfb_myh11",
        "cyto__apl_like_t_15_17_pml_rara",
        "cyto__t_9_11_mllt3_kmt2a",
        "cyto__kmt2a_rearranged_other",
        "cyto__t_6_9_dek_nup214",
        "cyto__t_9_22_bcr_abl1_like",
        "cyto__inv_3_or_t_3_3_or_3q26_meccom",
    ]
    recurrent_any = (f[recurrent_cols].sum(axis=1) > 0).to_numpy()
    f["cyto__complex_karyotype_strict"] = (
        (abn_count.to_numpy() >= 3) & (~recurrent_any) & (~pure_multi_trisomy.to_numpy())
    ).astype(float)
    f["cyto__monosomal_karyotype_strict"] = (
        ((monosomy_count.to_numpy() >= 2) | ((monosomy_count.to_numpy() == 1) & (structural_count.to_numpy() >= 1)))
        & (f["cyto__cbf_any"].to_numpy() == 0)
    ).astype(float)
    f["cyto__adverse_eln_cyto_any"] = (
        f[
            [
                "cyto__kmt2a_rearranged_other",
                "cyto__t_6_9_dek_nup214",
                "cyto__t_9_22_bcr_abl1_like",
                "cyto__inv_3_or_t_3_3_or_3q26_meccom",
                "cyto__chr5_abn",
                "cyto__chr7_abn",
                "cyto__chr17p_abn",
                "cyto__complex_karyotype_strict",
                "cyto__monosomal_karyotype_strict",
            ]
        ].sum(axis=1)
        > 0
    ).astype(float)
    f["cyto__intermediate_like_other_abnormality"] = (
        (texts.to_numpy() != "")
        & (f["cyto__normal_karyotype_strict"].to_numpy() == 0)
        & (f["cyto__cbf_any"].to_numpy() == 0)
        & (f["cyto__apl_like_t_15_17_pml_rara"].to_numpy() == 0)
        & (f["cyto__adverse_eln_cyto_any"].to_numpy() == 0)
    ).astype(float)
    f["cyto__monosomal_like_proxy"] = (
        (f["cyto__monosomal_karyotype_strict"].to_numpy() == 0)
        & (monosomy_count.to_numpy() >= 1)
        & (structural_count.to_numpy() >= 1)
    ).astype(float)
    return f


cyto_features_train = make_cytogenetic_features(clinical_train)
cyto_features_test = make_cytogenetic_features(clinical_test)
print("Cytogenetic feature count:", cyto_features_train.shape[1])


# %% [markdown]
# # Step 9: Molecular features
#
# Top genes and top effects are selected from training molecular data only.
# Test data is transformed with this fixed list.

# %%
KEY_GENES = [
    "TP53", "NPM1", "FLT3", "DNMT3A", "TET2", "ASXL1", "RUNX1", "SRSF2", "SF3B1",
    "U2AF1", "EZH2", "IDH1", "IDH2", "NRAS", "KRAS", "CBL", "JAK2", "BCOR",
    "STAG2", "WT1", "CEBPA", "DDX41", "ZRSR2", "ETV6", "GATA2", "KIT",
]

GENE_SETS = {
    "signaling": ["FLT3", "NRAS", "KRAS", "CBL", "JAK2", "KIT", "PTPN11"],
    "methylation": ["DNMT3A", "TET2", "IDH1", "IDH2"],
    "spliceosome": ["SRSF2", "SF3B1", "U2AF1", "ZRSR2"],
    "chromatin": ["ASXL1", "EZH2", "BCOR", "STAG2", "KMT2A"],
    "transcription": ["RUNX1", "CEBPA", "GATA2", "ETV6", "WT1"],
    "tumor_suppressor": ["TP53", "DDX41", "PHF6"],
}

TOP_N_GENES = 80
TOP_N_EFFECTS = 12

train_gene_counts = molecular_train["GENE"].fillna("UNKNOWN").astype(str).value_counts()
train_effect_counts = molecular_train["EFFECT"].fillna("UNKNOWN").astype(str).value_counts()

SELECTED_GENES = list(dict.fromkeys(train_gene_counts.head(TOP_N_GENES).index.tolist() + KEY_GENES))
SELECTED_EFFECTS = list(dict.fromkeys(train_effect_counts.head(TOP_N_EFFECTS).index.tolist()))

print("Selected genes from train/fixed clinical list:", len(SELECTED_GENES))
print("Selected effects from train:", SELECTED_EFFECTS)


def make_molecular_features(mol, patient_ids, selected_genes, selected_effects):
    patient_ids = pd.Index(pd.Series(patient_ids).astype(str), name="ID")
    m = mol.copy()
    m["ID"] = m["ID"].astype(str)
    m["GENE"] = m["GENE"].fillna("UNKNOWN").astype(str)
    m["EFFECT"] = m["EFFECT"].fillna("UNKNOWN").astype(str)
    m["VAF"] = pd.to_numeric(m["VAF"], errors="coerce")
    m["DEPTH"] = pd.to_numeric(m["DEPTH"], errors="coerce")
    m["REF"] = m["REF"].fillna("").astype(str)
    m["ALT"] = m["ALT"].fillna("").astype(str)
    m = m[m["ID"].isin(patient_ids)]

    f = pd.DataFrame(index=patient_ids)
    count = m.groupby("ID").size()
    f["Nmut"] = count.reindex(patient_ids, fill_value=0).astype(float)
    f["has_any_mutation"] = (f["Nmut"] > 0).astype(float)

    for col in ["VAF", "DEPTH"]:
        grp = m.groupby("ID")[col]
        f[f"{col}_mean"] = grp.mean().reindex(patient_ids)
        f[f"{col}_median"] = grp.median().reindex(patient_ids)
        f[f"{col}_max"] = grp.max().reindex(patient_ids)
        f[f"{col}_min"] = grp.min().reindex(patient_ids)
        f[f"{col}_std"] = grp.std().reindex(patient_ids)

    f["VAF_gt_0p10_count"] = m.assign(flag=(m["VAF"] > 0.10).astype(float)).groupby("ID")["flag"].sum().reindex(patient_ids, fill_value=0)
    f["VAF_gt_0p25_count"] = m.assign(flag=(m["VAF"] > 0.25).astype(float)).groupby("ID")["flag"].sum().reindex(patient_ids, fill_value=0)
    f["VAF_gt_0p40_count"] = m.assign(flag=(m["VAF"] > 0.40).astype(float)).groupby("ID")["flag"].sum().reindex(patient_ids, fill_value=0)
    f["VAF_lt_0p05_count"] = m.assign(flag=(m["VAF"] < 0.05).astype(float)).groupby("ID")["flag"].sum().reindex(patient_ids, fill_value=0)

    m["is_snv"] = ((m["REF"].str.len() == 1) & (m["ALT"].str.len() == 1)).astype(float)
    m["is_indel"] = (m["REF"].str.len() != m["ALT"].str.len()).astype(float)
    f["snv_count"] = m.groupby("ID")["is_snv"].sum().reindex(patient_ids, fill_value=0)
    f["indel_count"] = m.groupby("ID")["is_indel"].sum().reindex(patient_ids, fill_value=0)
    f["indel_fraction"] = f["indel_count"] / f["Nmut"].replace(0, np.nan)

    effect_counts = (
        m[m["EFFECT"].isin(selected_effects)]
        .pivot_table(index="ID", columns="EFFECT", values="GENE", aggfunc="size", fill_value=0)
        .reindex(index=patient_ids, columns=selected_effects, fill_value=0)
    )
    for effect in selected_effects:
        clean = re.sub(r"[^0-9a-zA-Z]+", "_", effect).strip("_")
        f[f"effect_count_{clean}"] = effect_counts[effect].astype(float)

    for gene in selected_genes:
        sub = m[m["GENE"] == gene]
        safe_gene = re.sub(r"[^0-9a-zA-Z]+", "_", gene).strip("_")
        gene_count = sub.groupby("ID").size().reindex(patient_ids, fill_value=0).astype(float)
        f[f"gene_{safe_gene}_count"] = gene_count
        f[f"gene_{safe_gene}_flag"] = (gene_count > 0).astype(float)
        f[f"gene_{safe_gene}_max_vaf"] = sub.groupby("ID")["VAF"].max().reindex(patient_ids)

    for set_name, genes in GENE_SETS.items():
        sub = m[m["GENE"].isin(genes)]
        set_count = sub.groupby("ID").size().reindex(patient_ids, fill_value=0).astype(float)
        f[f"pathway_{set_name}_count"] = set_count
        f[f"pathway_{set_name}_flag"] = (set_count > 0).astype(float)
        f[f"pathway_{set_name}_max_vaf"] = sub.groupby("ID")["VAF"].max().reindex(patient_ids)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


ELN9_MR_GENES = ["ASXL1", "BCOR", "EZH2", "RUNX1", "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2"]
WHO8_MR_GENES = ["ASXL1", "BCOR", "EZH2", "SF3B1", "SRSF2", "STAG2", "U2AF1", "ZRSR2"]
SPLICEOSOME_GENES = ["SF3B1", "SRSF2", "U2AF1", "ZRSR2"]
EPIGENETIC_GENES = ["DNMT3A", "TET2", "IDH1", "IDH2", "ASXL1", "EZH2"]
SIGNALING_GENES = ["FLT3", "KIT", "NRAS", "KRAS", "CBL", "JAK2"]
TRANSCRIPTION_FACTOR_GENES = ["RUNX1", "CEBPA", "ETV6", "WT1"]
HIGH_RISK_SUPPRESSOR_GENES = ["TP53", "WT1", "BCOR", "STAG2", "EZH2", "ETV6", "ZRSR2"]
KEY_VAF_GENES = [
    "TP53", "NPM1", "FLT3", "CEBPA", "ASXL1", "RUNX1", "EZH2", "BCOR", "STAG2",
    "SF3B1", "SRSF2", "U2AF1", "ZRSR2", "DNMT3A", "TET2", "IDH1", "IDH2",
    "NRAS", "KRAS", "KIT", "WT1", "CBL", "JAK2",
]


def prepare_molecular_rows(mol, patient_ids):
    patient_ids = pd.Index(pd.Series(patient_ids).astype(str), name="ID")
    m = mol.copy()
    m["ID"] = m["ID"].astype(str)
    m["GENE"] = m["GENE"].fillna("UNKNOWN").astype(str).str.upper()
    m["EFFECT"] = m["EFFECT"].fillna("").astype(str)
    m["PROTEIN_CHANGE"] = m["PROTEIN_CHANGE"].fillna("").astype(str)
    m["VAF"] = pd.to_numeric(m["VAF"], errors="coerce")
    for col in ["CHR", "START", "END", "REF", "ALT"]:
        m[col] = m[col].fillna("").astype(str)
    m = m[m["ID"].isin(patient_ids)].copy()
    key_cols = ["CHR", "START", "END", "REF", "ALT", "GENE", "PROTEIN_CHANGE", "EFFECT"]
    m["variant_key"] = m[key_cols].astype(str).agg("|".join, axis=1)
    return m.drop_duplicates(["ID", "variant_key"]), patient_ids


def is_nonsilent_effect(effect):
    e = str(effect).lower()
    positive = [
        "missense", "non_synonymous", "frameshift", "stop_gained", "stop_lost",
        "start_lost", "initiator_codon", "splice_acceptor", "splice_donor",
        "splice_site", "inframe_insertion", "inframe_deletion", "inframe_codon_gain",
        "inframe_codon_loss", "protein_altering", "itd", "ptd",
    ]
    negative = ["synonymous", "intronic", "upstream", "downstream", "utr"]
    return any(tok in e for tok in positive) and not (
        any(tok in e for tok in negative) and not any(tok in e for tok in positive[:-2])
    )


def aa_position_bounds(protein_change):
    nums = [int(x) for x in re.findall(r"\d+", str(protein_change))]
    if not nums:
        return None, None
    return min(nums), max(nums)


def intersects_range(lo, hi, start, end):
    if lo is None or hi is None:
        return False
    return max(lo, start) <= min(hi, end)


def gene_any(m, patient_ids, gene):
    return (m[m["GENE"] == gene].groupby("ID").size().reindex(patient_ids, fill_value=0) > 0).astype(float)


def nonsilent_gene_rows(m, gene=None, genes=None):
    sub = m[m["is_nonsilent"]].copy()
    if gene is not None:
        sub = sub[sub["GENE"] == gene]
    if genes is not None:
        sub = sub[sub["GENE"].isin(genes)]
    return sub


def unique_nonsilent_gene_count(m, patient_ids, genes):
    sub = nonsilent_gene_rows(m, genes=genes)
    if sub.empty:
        return pd.Series(0.0, index=patient_ids)
    return sub.groupby("ID")["GENE"].nunique().reindex(patient_ids, fill_value=0).astype(float)


def make_v7b_molecular_features(mol, patient_ids):
    m, patient_ids = prepare_molecular_rows(mol, patient_ids)
    m["is_nonsilent"] = m["EFFECT"].map(is_nonsilent_effect)
    protein_upper = m["PROTEIN_CHANGE"].str.upper()
    effect_upper = m["EFFECT"].str.upper()

    f = pd.DataFrame(index=patient_ids)

    npm1 = m[m["GENE"] == "NPM1"].copy()
    npm1_like = npm1[effect_upper.loc[npm1.index].str.contains("FRAMESHIFT", na=False) | protein_upper.loc[npm1.index].str.contains("FS", na=False)]
    f["mol__npm1_aml_like"] = (npm1_like.groupby("ID").size().reindex(patient_ids, fill_value=0) > 0).astype(float)

    flt3 = m[m["GENE"] == "FLT3"].copy()
    flt3_pos = flt3["PROTEIN_CHANGE"].map(aa_position_bounds)
    flt3_lo = flt3_pos.map(lambda x: x[0])
    flt3_hi = flt3_pos.map(lambda x: x[1])
    flt3_tkd = (
        protein_upper.loc[flt3.index].str.contains("D835|I836", na=False)
        | [intersects_range(lo, hi, 835, 836) for lo, hi in zip(flt3_lo, flt3_hi)]
    )
    f["mol__flt3_tkd_d835_i836"] = (
        flt3.loc[flt3_tkd].groupby("ID").size().reindex(patient_ids, fill_value=0) > 0
    ).astype(float)
    flt3_itd_explicit = effect_upper.loc[flt3.index].str.contains("ITD|INTERNAL TANDEM DUPLICATION", na=False) | protein_upper.loc[flt3.index].str.contains("ITD|INTERNAL TANDEM DUPLICATION", na=False)
    flt3_inframe_jm = (
        effect_upper.loc[flt3.index].str.contains("INFRAME|INSERTION|DUP|DELINS|CODON_GAIN", na=False)
        | protein_upper.loc[flt3.index].str.contains("INS|DUP|DELINS", na=False)
    ) & [intersects_range(lo, hi, 572, 610) for lo, hi in zip(flt3_lo, flt3_hi)]
    f["mol__flt3_itd_conservative"] = (
        flt3.loc[(flt3_itd_explicit | flt3_inframe_jm) & (~pd.Series(flt3_tkd, index=flt3.index).astype(bool))]
        .groupby("ID").size().reindex(patient_ids, fill_value=0) > 0
    ).astype(float)

    cebpa = m[m["GENE"] == "CEBPA"].copy()
    cebpa_pos = cebpa["PROTEIN_CHANGE"].map(aa_position_bounds)
    cebpa_in_bzip = [intersects_range(lo, hi, 272, 358) for lo, hi in cebpa_pos]
    cebpa_inframe = effect_upper.loc[cebpa.index].str.contains("INFRAME|CODON_GAIN|CODON_LOSS|PROTEIN_ALTERING", na=False)
    f["mol__cebpa_bzip_inframe"] = (
        cebpa.loc[cebpa_inframe & pd.Series(cebpa_in_bzip, index=cebpa.index)]
        .groupby("ID").size().reindex(patient_ids, fill_value=0) > 0
    ).astype(float)

    tp53 = nonsilent_gene_rows(m, gene="TP53")
    f["mol__tp53_any_nonsilent"] = (tp53.groupby("ID").size().reindex(patient_ids, fill_value=0) > 0).astype(float)
    f["vaf__tp53_max"] = tp53.groupby("ID")["VAF"].max().reindex(patient_ids, fill_value=0.0).fillna(0.0).astype(float)
    f["vaf__tp53_ge_0_10"] = (f["vaf__tp53_max"] >= 0.10).astype(float)
    f["count__tp53_variant_count"] = tp53.groupby("ID")["variant_key"].nunique().reindex(patient_ids, fill_value=0).astype(float)
    return f


def make_v7c_mr_helper_features(mol, patient_ids):
    m, patient_ids = prepare_molecular_rows(mol, patient_ids)
    m["is_nonsilent"] = m["EFFECT"].map(is_nonsilent_effect)
    f = pd.DataFrame(index=patient_ids)
    f["count__mr_gene_count_eln9"] = unique_nonsilent_gene_count(m, patient_ids, ELN9_MR_GENES)
    f["mol__mr_gene_any_eln9_nonsilent"] = (f["count__mr_gene_count_eln9"] > 0).astype(float)
    return f


def make_v7d_molecular_proxy_features(mol, patient_ids):
    m, patient_ids = prepare_molecular_rows(mol, patient_ids)
    m["is_nonsilent"] = m["EFFECT"].map(is_nonsilent_effect)
    f = pd.DataFrame(index=patient_ids)
    f["mol__mr_gene_any_who8_nonsilent"] = (unique_nonsilent_gene_count(m, patient_ids, WHO8_MR_GENES) > 0).astype(float)
    f["count__mr_gene_count_ge_2"] = (unique_nonsilent_gene_count(m, patient_ids, ELN9_MR_GENES) >= 2).astype(float)

    groups = {
        "spliceosome": SPLICEOSOME_GENES,
        "epigenetic": EPIGENETIC_GENES,
        "signaling": SIGNALING_GENES,
        "transcription_factor": TRANSCRIPTION_FACTOR_GENES,
        "high_risk_suppressor": HIGH_RISK_SUPPRESSOR_GENES,
    }
    for group_name, genes in groups.items():
        count = unique_nonsilent_gene_count(m, patient_ids, genes)
        f[f"count__group_gene_count__{group_name}"] = count
        f[f"mol__group_any__{group_name}"] = (count > 0).astype(float)

    nonsilent = m[m["is_nonsilent"]].copy()
    for gene in KEY_VAF_GENES:
        safe_gene = re.sub(r"[^0-9A-Za-z]+", "_", gene)
        f[f"vaf__max_vaf__{safe_gene}"] = (
            nonsilent[nonsilent["GENE"] == gene]
            .groupby("ID")["VAF"].max().reindex(patient_ids, fill_value=0.0).fillna(0.0).astype(float)
        )
    for threshold in [0.30, 0.40]:
        genes_over = nonsilent[(nonsilent["GENE"].isin(KEY_VAF_GENES)) & (nonsilent["VAF"] >= threshold)]
        name = f"count__key_gene_vaf_ge_0_{int(threshold * 100):02d}"
        f[name] = genes_over.groupby("ID")["GENE"].nunique().reindex(patient_ids, fill_value=0).astype(float)
    return f


molecular_features_train = make_molecular_features(
    molecular_train, clinical_train["ID"], SELECTED_GENES, SELECTED_EFFECTS
)
molecular_features_test = make_molecular_features(
    molecular_test, clinical_test["ID"], SELECTED_GENES, SELECTED_EFFECTS
)
print("Molecular feature count:", molecular_features_train.shape[1])


# %% [markdown]
# # Step 10: Assemble raw feature matrices
#
# No imputation, scaling, constant filtering, or duplicate detection here.
# Those steps happen fold-locally inside `FoldPreprocessor`.

# %%
def assemble_features(clinical_features, cyto_features, molecular_features):
    f = pd.concat([clinical_features, cyto_features, molecular_features], axis=1)

    def get_col(name, default=0.0):
        if name in f.columns:
            return f[name].fillna(0)
        return pd.Series(default, index=f.index)

    # Fixed clinically motivated interactions. These use already-built features only.
    f["interaction_TP53_complex"] = get_col("gene_TP53_flag") * get_col("cyto_complex_3plus")
    f["interaction_TP53_del17p"] = get_col("gene_TP53_flag") * get_col("cyto_del17p")
    f["interaction_NPM1_FLT3"] = get_col("gene_NPM1_flag") * get_col("gene_FLT3_flag")
    f["interaction_ASXL1_RUNX1"] = get_col("gene_ASXL1_flag") * get_col("gene_RUNX1_flag")
    f["interaction_spliceosome_anemia"] = get_col("pathway_spliceosome_flag") * get_col("HB_lt_10")
    f["interaction_signaling_high_wbc"] = get_col("pathway_signaling_flag") * get_col("WBC_ge_10")
    f["interaction_methylation_high_nmut"] = get_col("pathway_methylation_flag") * (get_col("Nmut") >= 4).astype(float)

    f = f.apply(pd.to_numeric, errors="coerce")
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def make_v7c_interaction_features(features):
    def get_col(name, default=0.0):
        if name in features.columns:
            return features[name].fillna(0)
        return pd.Series(default, index=features.index)

    f = pd.DataFrame(index=features.index)
    f["int__tp53_multihit_proxy"] = (
        (get_col("count__tp53_variant_count") >= 2)
        | (get_col("vaf__tp53_max") >= 0.50)
        | (
            (get_col("vaf__tp53_ge_0_10") == 1)
            & ((get_col("cyto__chr17p_abn") == 1) | (get_col("cyto__complex_karyotype_strict") == 1))
        )
    ).astype(float)
    f["int__npm1_without_flt3_itd"] = (
        (get_col("mol__npm1_aml_like") == 1) & (get_col("mol__flt3_itd_conservative") == 0)
    ).astype(float)
    f["int__npm1_with_flt3_itd"] = (
        (get_col("mol__npm1_aml_like") == 1) & (get_col("mol__flt3_itd_conservative") == 1)
    ).astype(float)
    f["int__npm1_with_adverse_cyto"] = (
        (get_col("mol__npm1_aml_like") == 1) & (get_col("cyto__adverse_eln_cyto_any") == 1)
    ).astype(float)
    f["int__tp53_with_complex_karyotype"] = (
        (get_col("mol__tp53_any_nonsilent") == 1) & (get_col("cyto__complex_karyotype_strict") == 1)
    ).astype(float)
    f["int__tp53_with_chr17p_abn"] = (
        (get_col("mol__tp53_any_nonsilent") == 1) & (get_col("cyto__chr17p_abn") == 1)
    ).astype(float)
    f["int__adverse_cyto_with_mr_gene"] = (
        (get_col("cyto__adverse_eln_cyto_any") == 1)
        & (get_col("mol__mr_gene_any_eln9_nonsilent") == 1)
    ).astype(float)
    f["int__cbf_with_kit"] = (
        (get_col("cyto__cbf_any") == 1) & (get_col("gene_KIT_flag") == 1)
    ).astype(float)
    f["int__cbf_with_flt3_itd"] = (
        (get_col("cyto__cbf_any") == 1) & (get_col("mol__flt3_itd_conservative") == 1)
    ).astype(float)
    ras_any = (
        (get_col("gene_NRAS_flag") == 1)
        | (get_col("gene_KRAS_flag") == 1)
        | (get_col("gene_CBL_flag") == 1)
    )
    f["int__cbf_with_ras_pathway"] = ((get_col("cyto__cbf_any") == 1) & ras_any).astype(float)

    apl = get_col("cyto__apl_like_t_15_17_pml_rara") == 1
    adverse = (
        (get_col("cyto__adverse_eln_cyto_any") == 1)
        | (get_col("vaf__tp53_ge_0_10") == 1)
        | (
            (get_col("mol__mr_gene_any_eln9_nonsilent") == 1)
            & (get_col("cyto__cbf_any") == 0)
            & (get_col("mol__cebpa_bzip_inframe") == 0)
            & (f["int__npm1_without_flt3_itd"] == 0)
            & (get_col("cyto__t_9_11_mllt3_kmt2a") == 0)
        )
    ) & (~apl)
    favorable = (
        (
            (get_col("cyto__cbf_any") == 1)
            | (get_col("mol__cebpa_bzip_inframe") == 1)
            | (f["int__npm1_without_flt3_itd"] == 1)
        )
        & (~apl)
        & (~adverse)
    )
    intermediate = (
        (
            (f["int__npm1_with_flt3_itd"] == 1)
            | (
                (get_col("mol__flt3_itd_conservative") == 1)
                & (get_col("mol__npm1_aml_like") == 0)
                & (get_col("cyto__adverse_eln_cyto_any") == 0)
            )
            | (get_col("cyto__t_9_11_mllt3_kmt2a") == 1)
        )
        & (~apl)
        & (~adverse)
        & (~favorable)
    )
    f["risk__eln2022_like_favorable"] = favorable.astype(float)
    f["risk__eln2022_like_intermediate"] = intermediate.astype(float)
    f["risk__eln2022_like_adverse"] = adverse.astype(float)
    return f


feature_train_v6 = assemble_features(clinical_features_train, cyto_features_train, molecular_features_train)
feature_test_v6 = assemble_features(clinical_features_test, cyto_features_test, molecular_features_test)

def clean_numeric_feature_frame(f, label):
    f = f.apply(pd.to_numeric, errors="coerce")
    f = f.replace([np.inf, -np.inf], np.nan)
    if f.columns.duplicated().any():
        duplicates = f.columns[f.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate raw feature names in {label}: {duplicates[:10]}")
    return f


def build_v6_features_only():
    feature_train = clean_numeric_feature_frame(feature_train_v6.copy(), "v8a/v6/train")
    feature_test = clean_numeric_feature_frame(feature_test_v6.copy(), "v8a/v6/test").reindex(columns=feature_train.columns)
    X_train_v6 = feature_train.loc[train_ids].copy()
    X_test_v6 = feature_test.loc[test_ids].copy()

    assert "CENTER" not in X_train_v6.columns
    assert "CENTER" not in X_test_v6.columns
    assert "CYTOGENETICS" not in X_train_v6.columns
    assert "CYTOGENETICS" not in X_test_v6.columns
    assert list(X_test_v6.index) == list(random_submission["ID"])
    assert list(X_train_v6.columns) == list(X_test_v6.columns)
    assert not X_train_v6.columns.duplicated().any()
    assert not X_test_v6.columns.duplicated().any()
    return X_train_v6, X_test_v6


X_train_raw, X_test_raw = build_v6_features_only()
feature_counts_by_phase = pd.DataFrame([
    {
        "feature_set": "v6_only",
        "train_rows": X_train_raw.shape[0],
        "train_features": X_train_raw.shape[1],
        "test_rows": X_test_raw.shape[0],
        "test_features": X_test_raw.shape[1],
        "train_missing_pct": round(float(X_train_raw.isna().mean().mean() * 100), 4),
        "center_absent": "CENTER" not in X_train_raw.columns and "CENTER" not in X_test_raw.columns,
        "cytogenetics_absent": "CYTOGENETICS" not in X_train_raw.columns and "CYTOGENETICS" not in X_test_raw.columns,
        "train_test_columns_match": list(X_train_raw.columns) == list(X_test_raw.columns),
    }
])
display(feature_counts_by_phase)
feature_counts_by_phase.to_csv(OUTPUT_DIR / "v8a_feature_counts.csv", index=False)

print("V8a V6-only train shape:", X_train_raw.shape)
print("V8a V6-only test shape:", X_test_raw.shape)
print("CENTER absent:", "CENTER" not in X_train_raw.columns and "CENTER" not in X_test_raw.columns)
print("CYTOGENETICS absent:", "CYTOGENETICS" not in X_train_raw.columns and "CYTOGENETICS" not in X_test_raw.columns)
print("train/test columns match:", list(X_train_raw.columns) == list(X_test_raw.columns))

if FEATURE_COUNTS_ONLY:
    no_submission_path = OUTPUT_DIR / "v8a_no_submission_summary.csv"
    feature_counts_by_phase.to_csv(no_submission_path, index=False)
    print("FEATURE_COUNTS_ONLY=1: skipping full V8a GroupKFold run.")
    print("No submission files generated.")
    sys.exit(0)


# %% [markdown]
# # Step 11: Fold-local preprocessing
#
# Safety rules:
# - all-missing columns are detected and removed using only `X_tr`;
# - imputer is fit only on `X_tr`;
# - nonconstant filtering is based only on imputed `X_tr`;
# - duplicate detection is based only on imputed/filtered `X_tr`;
# - validation/test reuse the exact kept-column list.

# %%
class FoldPreprocessor:
    def fit(self, X):
        assert "CENTER" not in X.columns
        self.input_columns_ = X.columns.tolist()

        self.kept_not_all_missing_ = X.columns[~X.isna().all(axis=0)].tolist()
        if len(self.kept_not_all_missing_) == 0:
            raise ValueError("All columns are all-missing in this training fold.")

        X0 = X[self.kept_not_all_missing_].copy()

        self.imputer_ = SimpleImputer(strategy="median")
        Xi = pd.DataFrame(
            self.imputer_.fit_transform(X0),
            columns=self.kept_not_all_missing_,
            index=X.index,
        )

        self.kept_nonconstant_ = Xi.columns[Xi.nunique(dropna=False) > 1].tolist()
        if len(self.kept_nonconstant_) == 0:
            raise ValueError("No nonconstant columns remain after fold-local imputation.")
        Xi = Xi[self.kept_nonconstant_]

        self.kept_columns_ = Xi.T.drop_duplicates().T.columns.tolist()
        if len(self.kept_columns_) == 0:
            raise ValueError("No columns remain after fold-local duplicate removal.")

        self.scaler_ = StandardScaler()
        self.scaler_.fit(Xi[self.kept_columns_])
        return self

    def transform(self, X):
        assert "CENTER" not in X.columns
        X0 = X[self.kept_not_all_missing_].copy()
        Xi = pd.DataFrame(
            self.imputer_.transform(X0),
            columns=self.kept_not_all_missing_,
            index=X.index,
        )
        Xi = Xi[self.kept_columns_]
        Xi = pd.DataFrame(
            self.scaler_.transform(Xi),
            columns=self.kept_columns_,
            index=X.index,
        )
        assert np.isfinite(Xi.to_numpy()).all()
        return Xi


def preprocess_fold(X_tr, X_va):
    pp = FoldPreprocessor().fit(X_tr)
    return pp, pp.transform(X_tr), pp.transform(X_va)


# %% [markdown]
# # Step 12: CV, scoring, and utility helpers

# %%
def ipcw_score(y_train_fold, y_valid_fold, pred_valid, tau=TAU):
    return float(concordance_index_ipcw(y_train_fold, y_valid_fold, pred_valid, tau=tau)[0])


def rank01(values):
    values = np.asarray(values, dtype=float)
    return pd.Series(values).rank(method="average").to_numpy(dtype=float) / (len(values) + 1.0)


def make_group_splits(groups, n_splits=5):
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(X_train_raw, y, groups=groups))


def make_random_splits(n_splits=5):
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X_train_raw, y))


def make_loco_splits(groups, min_valid_size=20):
    groups = np.asarray(groups)
    splits = []
    for center in sorted(pd.Series(groups).dropna().unique()):
        valid_idx = np.flatnonzero(groups == center)
        if len(valid_idx) < min_valid_size:
            continue
        train_idx = np.flatnonzero(groups != center)
        splits.append((train_idx, valid_idx, center))
    return splits


def validate_submission_frame(submission_df):
    assert list(submission_df.columns) == ["ID", "risk_score"]
    assert len(submission_df) == len(random_submission)
    assert list(submission_df["ID"]) == list(random_submission["ID"])
    assert np.isfinite(submission_df["risk_score"]).all()


def fit_predict_model(model, X_tr_pp, y_tr, X_va_pp, X_test_pp=None):
    model.fit(X_tr_pp, y_tr)
    pred_va = np.asarray(model.predict(X_va_pp), dtype=float)
    if pred_va.ndim != 1:
        pred_va = pred_va.reshape(-1)
    if not np.isfinite(pred_va).all():
        raise ValueError("non-finite validation predictions")

    pred_test = None
    if X_test_pp is not None:
        pred_test = np.asarray(model.predict(X_test_pp), dtype=float)
        if pred_test.ndim != 1:
            pred_test = pred_test.reshape(-1)
        if not np.isfinite(pred_test).all():
            raise ValueError("non-finite test predictions")
    return pred_va, pred_test


def run_cv_for_model(
    model_name,
    model,
    X_raw,
    y_fit_all,
    y_score_all,
    splits,
    X_test_raw=None,
    split_labels=None,
    require_complete_oof=True,
):
    assert "CENTER" not in X_raw.columns
    if X_test_raw is not None:
        assert "CENTER" not in X_test_raw.columns

    start = time.time()
    oof_pred = np.full(len(X_raw), np.nan, dtype=float)
    oof_rank = np.full(len(X_raw), np.nan, dtype=float)
    fold_scores = []
    fold_test_ranks = []
    fold_details = []

    for fold, split in enumerate(splits, start=1):
        if len(split) == 3:
            tr_idx, va_idx, label = split
        else:
            tr_idx, va_idx = split
            label = fold if split_labels is None else split_labels[fold - 1]

        try:
            X_tr_raw = X_raw.iloc[tr_idx]
            X_va_raw = X_raw.iloc[va_idx]
            pp, X_tr_pp, X_va_pp = preprocess_fold(X_tr_raw, X_va_raw)
            X_test_pp = pp.transform(X_test_raw) if X_test_raw is not None else None

            fitted = clone(model)
            pred_va, pred_test = fit_predict_model(fitted, X_tr_pp, y_fit_all[tr_idx], X_va_pp, X_test_pp)
            score = ipcw_score(y_score_all[tr_idx], y_score_all[va_idx], pred_va)

            oof_pred[va_idx] = pred_va
            oof_rank[va_idx] = rank01(pred_va)
            fold_scores.append(score)
            fold_details.append({"fold": fold, "label": label, "score": score, "n_valid": len(va_idx)})

            if pred_test is not None:
                fold_test_ranks.append(rank01(pred_test))

            print(f"  {model_name} fold {fold} ({label}) IPCW={score:.5f} n_valid={len(va_idx)}")

        except Exception as exc:
            print(f"  Skipping {model_name} fold {fold} ({label}) after error: {repr(exc)}")
            continue
        finally:
            gc.collect()

    if len(fold_scores) == 0:
        raise ValueError(f"No successful folds for {model_name}.")
    if require_complete_oof and np.isnan(oof_pred).any():
        missing = int(np.isnan(oof_pred).sum())
        raise ValueError(f"{model_name} has {missing} missing OOF predictions after CV.")

    test_rank = None
    if X_test_raw is not None:
        if len(fold_test_ranks) == 0:
            raise ValueError(f"No successful test predictions for {model_name}.")
        test_rank = np.mean(np.vstack(fold_test_ranks), axis=0)
        if not np.isfinite(test_rank).all():
            raise ValueError(f"{model_name} has non-finite averaged test ranks.")

    elapsed = time.time() - start
    return {
        "model_name": model_name,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "oof_pred": oof_pred,
        "oof_rank": oof_rank,
        "test_rank": test_rank,
        "fold_details": fold_details,
        "elapsed_sec": elapsed,
    }


def run_guarded_cv(
    model_dict,
    splits,
    label,
    y_fit_all,
    y_score_all,
    X_test_for_prediction=None,
    require_complete_oof=True,
    require_success=True,
):
    successful_results = {}
    print(f"\n========== {label} ==========")

    for model_name, model in model_dict.items():
        try:
            print(f"\nRunning {model_name} for {label}...")
            result = run_cv_for_model(
                model_name=model_name,
                model=model,
                X_raw=X_train_raw,
                y_fit_all=y_fit_all,
                y_score_all=y_score_all,
                splits=splits,
                X_test_raw=X_test_for_prediction,
                require_complete_oof=require_complete_oof,
            )

            if not np.isfinite(result["mean_score"]):
                print(f"Skipping {model_name}: non-finite mean CV score")
                continue
            if require_complete_oof and not np.isfinite(result["oof_pred"]).all():
                print(f"Skipping {model_name}: non-finite OOF predictions")
                continue
            if result["test_rank"] is not None and not np.isfinite(result["test_rank"]).all():
                print(f"Skipping {model_name}: non-finite test predictions")
                continue

            successful_results[model_name] = result
            print(
                f"{model_name} {label}: mean={result['mean_score']:.5f} "
                f"std={result['std_score']:.5f} elapsed={result['elapsed_sec'] / 60:.1f} min"
            )

        except Exception as exc:
            print(f"Skipping {model_name} after error in {label}: {repr(exc)}")
            continue

    if require_success:
        assert len(successful_results) > 0, f"No models completed successfully for {label}."
    elif len(successful_results) == 0:
        print(f"No models completed successfully for {label}; continuing because this block is diagnostic only.")
    return successful_results


# %% [markdown]
# # Step 13: Model zoo
#
# `ExtraSurvivalTrees` is optional. Every model is guarded in CV and final fitting.

# %%
models = {
    "coxph": CoxPHSurvivalAnalysis(alpha=0.1),
    "coxnet": CoxnetSurvivalAnalysis(l1_ratio=0.2, alpha_min_ratio=0.01, n_alphas=50),
    "gbsa": GradientBoostingSurvivalAnalysis(
        random_state=RANDOM_STATE,
        n_estimators=350,
        learning_rate=0.03,
        max_depth=2,
    ),
    "rsf": RandomSurvivalForest(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
}

if HAS_EXTRA_SURVIVAL_TREES and ENABLE_EXTRA_SURVIVAL_TREES:
    models["extra"] = ExtraSurvivalTrees(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

print("Models:", list(models))


# %% [markdown]
# # Step 14: V8a GroupKFold target comparison harness
#
# V8a model and ensemble selection uses GroupKFold by CENTER only. Random KFold,
# LOCO, KYW diagnostics, prior submissions, and leaderboard feedback are not
# used for target selection, model selection, or final submission selection.

# %%
def score_oof_prediction(oof_values):
    if not np.isfinite(oof_values).all():
        raise ValueError("OOF values contain non-finite values.")
    return float(concordance_index_ipcw(y_full, y_full, oof_values, tau=TAU)[0])


def normalize_weight_dict(weight_dict):
    total = float(sum(weight_dict.values()))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid weights.")
    return {name: float(weight) / total for name, weight in weight_dict.items()}


def evaluate_target_ensembles(target_name, group_results):
    group_summary = pd.DataFrame([
        {
            "target": target_name,
            "model": name,
            "group_mean_ipcw": res["mean_score"],
            "group_std": res["std_score"],
            "elapsed_min": res["elapsed_sec"] / 60,
        }
        for name, res in group_results.items()
    ]).sort_values("group_mean_ipcw", ascending=False)
    eligible_models = group_summary["model"].tolist()
    best_single_model = eligible_models[0]
    top3_models = eligible_models[: min(3, len(eligible_models))]

    candidates = [
        ("best_single_groupcv", {best_single_model: 1.0}),
        ("top3_equal_rank", {name: 1.0 / len(top3_models) for name in top3_models}),
    ]
    top3_scores = {name: group_results[name]["mean_score"] for name in top3_models}
    min_score = min(top3_scores.values())
    score_weights = {name: max(score - min_score + 1e-4, 1e-4) for name, score in top3_scores.items()}
    candidates.append(("top3_score_weighted_rank", normalize_weight_dict(score_weights)))

    ensemble_rows = []
    for candidate_name, weights in candidates:
        weights = normalize_weight_dict(weights)
        ensemble_oof = np.zeros(len(X_train_raw), dtype=float)
        for model_name, weight in weights.items():
            ensemble_oof += weight * group_results[model_name]["oof_rank"]
        ensemble_score = score_oof_prediction(ensemble_oof)
        ensemble_rows.append(
            {
                "target": target_name,
                "candidate": candidate_name,
                "group_oof_ipcw": ensemble_score,
                "weights": weights,
                "model_names": list(weights.keys()),
                "ensemble_oof": ensemble_oof,
            }
        )
    ensemble_summary = pd.DataFrame(ensemble_rows).sort_values("group_oof_ipcw", ascending=False)
    selected = ensemble_summary.iloc[0].to_dict()
    return group_summary, ensemble_summary, selected


def score_subset_ipcw(train_idx, valid_idx, pred):
    if len(valid_idx) < 20:
        return np.nan
    try:
        return ipcw_score(y_full[train_idx], y_full[valid_idx], pred[valid_idx])
    except Exception:
        return np.nan


def target_delta_rows(target_name, target_oof, full_oof, splits):
    rows = []
    centers = np.asarray(groups_center)
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        target_fold_score = score_subset_ipcw(train_idx, valid_idx, target_oof)
        full_fold_score = score_subset_ipcw(train_idx, valid_idx, full_oof)
        rows.append(
            {
                "target": target_name,
                "fold": fold,
                "center": "__FOLD_ALL__",
                "n_valid": len(valid_idx),
                "full_ipcw": full_fold_score,
                "target_ipcw": target_fold_score,
                "delta_ipcw": target_fold_score - full_fold_score,
            }
        )
        for center in sorted(pd.Series(centers[valid_idx]).dropna().unique()):
            center_idx = valid_idx[centers[valid_idx] == center]
            target_center_score = score_subset_ipcw(train_idx, center_idx, target_oof)
            full_center_score = score_subset_ipcw(train_idx, center_idx, full_oof)
            rows.append(
                {
                    "target": target_name,
                    "fold": fold,
                    "center": center,
                    "n_valid": len(center_idx),
                    "full_ipcw": full_center_score,
                    "target_ipcw": target_center_score,
                    "delta_ipcw": target_center_score - full_center_score,
                }
            )
    return rows


def acceptance_stats(target_name, selected, full_selected, delta_df):
    target_score = selected["group_oof_ipcw"]
    full_score = full_selected["group_oof_ipcw"]
    scorable_centers = delta_df[(delta_df["target"] == target_name) & (delta_df["center"] != "__FOLD_ALL__")].dropna(subset=["delta_ipcw"])
    median_center_delta = float(scorable_centers["delta_ipcw"].median()) if len(scorable_centers) else np.nan
    pct_centers_improved = float((scorable_centers["delta_ipcw"] > 0).mean()) if len(scorable_centers) else np.nan
    v6_corrected_baseline = 0.7140458961158811
    v7b_corrected_baseline = 0.7152546075843262
    acceptance_threshold = v7b_corrected_baseline + 0.005
    accepted = (
        (target_score >= acceptance_threshold)
        and np.isfinite(median_center_delta)
        and (median_center_delta > 0)
        and np.isfinite(pct_centers_improved)
        and (pct_centers_improved >= 0.60)
    )
    return {
        "target": target_name,
        "selected_candidate": selected["candidate"],
        "selected_group_oof_ipcw": target_score,
        "full_target_group_oof_ipcw": full_score,
        "delta_vs_full_target": target_score - full_score,
        "delta_vs_v6_corrected": target_score - v6_corrected_baseline,
        "delta_vs_v7b_corrected": target_score - v7b_corrected_baseline,
        "acceptance_threshold": acceptance_threshold,
        "median_center_delta": median_center_delta,
        "pct_centers_improved": pct_centers_improved,
        "accepted": bool(accepted),
        "weights": selected["weights"],
        "models": selected["model_names"],
    }


group_splits = make_group_splits(groups_center, n_splits=5)
target_results = {}
target_group_summaries = []
target_ensemble_summaries = []
foldwise_model_rows = []
all_delta_rows = []

for target_name in TARGET_ORDER:
    print(f"\n================ V8a target comparison: {target_name} ================")
    y_fit = TARGET_CONFIGS[target_name]
    assert "CENTER" not in X_train_raw.columns
    assert "CENTER" not in X_test_raw.columns
    group_results = run_guarded_cv(
        models,
        group_splits,
        label=f"{target_name} target GroupKFold by CENTER selection",
        y_fit_all=y_fit,
        y_score_all=y_full,
        X_test_for_prediction=None,
    )
    group_summary, ensemble_summary, selected = evaluate_target_ensembles(target_name, group_results)
    display(group_summary)
    display(ensemble_summary[["target", "candidate", "group_oof_ipcw", "weights"]])
    target_results[target_name] = {
        "group_results": group_results,
        "group_summary": group_summary,
        "ensemble_summary": ensemble_summary,
        "selected": selected,
    }
    target_group_summaries.append(group_summary)
    target_ensemble_summaries.append(ensemble_summary.drop(columns=["ensemble_oof"]))
    for model_name, res in group_results.items():
        for detail in res["fold_details"]:
            foldwise_model_rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "fold": detail["fold"],
                    "fold_label": detail["label"],
                    "n_valid": detail["n_valid"],
                    "ipcw": detail["score"],
                }
            )

full_selected = target_results["full"]["selected"]
for target_name in TARGET_ORDER[1:]:
    selected = target_results[target_name]["selected"]
    all_delta_rows.extend(
        target_delta_rows(
            target_name,
            selected["ensemble_oof"],
            full_selected["ensemble_oof"],
            group_splits,
        )
    )

delta_df = pd.DataFrame(all_delta_rows)
comparison_rows = []
for target_name in TARGET_ORDER:
    selected = target_results[target_name]["selected"]
    if target_name == "full":
        comparison_rows.append(
            {
                "target": "full",
                "selected_candidate": selected["candidate"],
                "selected_group_oof_ipcw": selected["group_oof_ipcw"],
                "full_target_group_oof_ipcw": selected["group_oof_ipcw"],
                "delta_vs_full_target": 0.0,
                "delta_vs_v6_corrected": selected["group_oof_ipcw"] - 0.7140458961158811,
                "delta_vs_v7b_corrected": selected["group_oof_ipcw"] - 0.7152546075843262,
                "acceptance_threshold": 0.7202546075843262,
                "median_center_delta": 0.0,
                "pct_centers_improved": 0.0,
                "accepted": False,
                "weights": selected["weights"],
                "models": selected["model_names"],
            }
        )
    else:
        comparison_rows.append(acceptance_stats(target_name, selected, full_selected, delta_df))

target_comparison = pd.DataFrame(comparison_rows)
display(target_comparison)
pd.concat(target_group_summaries, ignore_index=True).to_csv(OUTPUT_DIR / "v8a_target_groupkfold_model_scores.csv", index=False)
pd.concat(target_ensemble_summaries, ignore_index=True).to_csv(OUTPUT_DIR / "v8a_target_groupkfold_ensemble_scores.csv", index=False)
pd.DataFrame(foldwise_model_rows).to_csv(OUTPUT_DIR / "v8a_foldwise_model_scores.csv", index=False)
delta_df.to_csv(OUTPUT_DIR / "v8a_foldwise_target_deltas.csv", index=False)
target_comparison.to_csv(OUTPUT_DIR / "v8a_target_comparison_summary.csv", index=False)

no_submission_path = OUTPUT_DIR / "v8a_no_submission_summary.csv"
target_comparison.to_csv(no_submission_path, index=False)
print("SAVE_V8_SUBMISSIONS=0: no V8a submission CSVs are generated by this script.")
print("Saved diagnostic summary:", no_submission_path)
