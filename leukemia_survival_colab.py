# %% [markdown]
# # Step 1: Colab setup
#
# This notebook builds leakage-safe survival-risk submissions for the leukemia
# challenge. It is written for Google Colab, but it also works as a Python
# script when the same packages and files are available locally.
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


RUN_RANDOM_KFOLD_DIAGNOSTIC = env_flag("RUN_RANDOM_KFOLD_DIAGNOSTIC", True)
RUN_LOCO_DIAGNOSTIC = env_flag("RUN_LOCO_DIAGNOSTIC", True)
ENABLE_EXTRA_SURVIVAL_TREES = env_flag("ENABLE_EXTRA_SURVIVAL_TREES", True)

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
]) / "qrt_outputs"
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
target_clean["OS_STATUS"] = target_clean["OS_STATUS"].astype(bool)
target_clean["OS_YEARS"] = target_clean["OS_YEARS"].astype(float)

train_ids = target_clean["ID"].tolist()
test_ids = random_submission["ID"].tolist()

groups_center = clinical_train.set_index("ID").loc[train_ids, "CENTER"].values
y = Surv.from_dataframe("OS_STATUS", "OS_YEARS", target_clean)

print("Clean training rows:", len(train_ids))
print("Event rate:", target_clean["OS_STATUS"].mean().round(4))
print("Max OS_YEARS:", target_clean["OS_YEARS"].max())


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


feature_train = assemble_features(clinical_features_train, cyto_features_train, molecular_features_train)
feature_test = assemble_features(clinical_features_test, cyto_features_test, molecular_features_test)
feature_test = feature_test.reindex(columns=feature_train.columns)

X_train_raw = feature_train.loc[train_ids].copy()
X_test_raw = feature_test.loc[test_ids].copy()

assert "CENTER" not in X_train_raw.columns
assert "CENTER" not in X_test_raw.columns
assert "CYTOGENETICS" not in X_train_raw.columns
assert "CYTOGENETICS" not in X_test_raw.columns
assert list(X_test_raw.index) == list(random_submission["ID"])
assert list(X_train_raw.columns) == list(X_test_raw.columns)
assert not X_train_raw.columns.duplicated().any()
assert not X_test_raw.columns.duplicated().any()

print("Raw train matrix:", X_train_raw.shape)
print("Raw test matrix:", X_test_raw.shape)
print("Raw train missing pct:", round(float(X_train_raw.isna().mean().mean() * 100), 2))


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
    y_all,
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
            pred_va, pred_test = fit_predict_model(fitted, X_tr_pp, y_all[tr_idx], X_va_pp, X_test_pp)
            score = ipcw_score(y_all[tr_idx], y_all[va_idx], pred_va)

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
                y_all=y,
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
        n_jobs=-1,
    ),
}

if HAS_EXTRA_SURVIVAL_TREES and ENABLE_EXTRA_SURVIVAL_TREES:
    models["extra"] = ExtraSurvivalTrees(
        n_estimators=500,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

print("Models:", list(models))


# %% [markdown]
# # Step 14: Random KFold diagnostic only
#
# These scores are never used for model selection, weights, or final submission choice.

# %%
if RUN_RANDOM_KFOLD_DIAGNOSTIC:
    random_splits = make_random_splits(n_splits=5)
    random_diagnostic_results = run_guarded_cv(
        models,
        random_splits,
        label="Random KFold diagnostic only",
        X_test_for_prediction=None,
    )
    random_diag_summary = pd.DataFrame([
        {
            "model": name,
            "random_kfold_mean_ipcw": res["mean_score"],
            "random_kfold_std": res["std_score"],
        }
        for name, res in random_diagnostic_results.items()
    ]).sort_values("random_kfold_mean_ipcw", ascending=False)
    display(random_diag_summary)
else:
    print("Random KFold diagnostic skipped by toggle.")


# %% [markdown]
# # Step 15: GroupKFold by CENTER for model and ensemble selection
#
# This is the only validation source used for model selection and ensemble weights.
# `CENTER` is passed only to the splitter and never included in feature matrices.

# %%
assert "CENTER" not in X_train_raw.columns
assert "CENTER" not in X_test_raw.columns

group_splits = make_group_splits(groups_center, n_splits=5)
group_results = run_guarded_cv(
    models,
    group_splits,
    label="GroupKFold by CENTER selection",
    X_test_for_prediction=X_test_raw,
)

group_summary = pd.DataFrame([
    {
        "model": name,
        "group_mean_ipcw": res["mean_score"],
        "group_std": res["std_score"],
        "elapsed_min": res["elapsed_sec"] / 60,
    }
    for name, res in group_results.items()
]).sort_values("group_mean_ipcw", ascending=False)
display(group_summary)


# %% [markdown]
# # Step 16: LOCO diagnostic only
#
# LOCO cannot override GroupKFold model selection, ensemble weights, or final submission.

# %%
if RUN_LOCO_DIAGNOSTIC:
    top_group_models_for_loco = group_summary["model"].head(min(3, len(group_summary))).tolist()
    loco_splits = make_loco_splits(groups_center, min_valid_size=20)
    loco_models = {name: models[name] for name in top_group_models_for_loco if name in models}
    loco_diagnostic_results = run_guarded_cv(
        loco_models,
        loco_splits,
        label="LOCO diagnostic only",
        X_test_for_prediction=None,
        require_complete_oof=False,
        require_success=False,
    )
    if loco_diagnostic_results:
        loco_diag_summary = pd.DataFrame([
            {
                "model": name,
                "loco_mean_ipcw": res["mean_score"],
                "loco_std": res["std_score"],
            }
            for name, res in loco_diagnostic_results.items()
        ]).sort_values("loco_mean_ipcw", ascending=False)
        display(loco_diag_summary)
else:
    print("LOCO diagnostic skipped by toggle.")


# %% [markdown]
# # Step 17: GroupKFold OOF rank ensembling
#
# Candidate ensemble weights are selected using GroupKFold OOF predictions only.

# %%
def score_oof_prediction(oof_values):
    if not np.isfinite(oof_values).all():
        raise ValueError("OOF values contain non-finite values.")
    return float(concordance_index_ipcw(y, y, oof_values, tau=TAU)[0])


def normalize_weight_dict(weight_dict):
    total = float(sum(weight_dict.values()))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid weights.")
    return {name: float(weight) / total for name, weight in weight_dict.items()}


eligible_models = group_summary["model"].tolist()
best_single_model = eligible_models[0]
top3_models = eligible_models[: min(3, len(eligible_models))]

candidate_weight_dicts = []
candidate_weight_dicts.append(("best_single_groupcv", {best_single_model: 1.0}))
candidate_weight_dicts.append(("top3_equal_rank", {name: 1.0 / len(top3_models) for name in top3_models}))

top3_scores = {name: group_results[name]["mean_score"] for name in top3_models}
min_score = min(top3_scores.values())
score_weights = {name: max(score - min_score + 1e-4, 1e-4) for name, score in top3_scores.items()}
candidate_weight_dicts.append(("top3_score_weighted_rank", normalize_weight_dict(score_weights)))

ensemble_rows = []
for candidate_name, weights in candidate_weight_dicts:
    weights = normalize_weight_dict(weights)
    ensemble_oof = np.zeros(len(X_train_raw), dtype=float)
    for model_name, weight in weights.items():
        ensemble_oof += weight * group_results[model_name]["oof_rank"]
    ensemble_score = score_oof_prediction(ensemble_oof)
    ensemble_rows.append({
        "candidate": candidate_name,
        "group_oof_ipcw": ensemble_score,
        "weights": weights,
    })

ensemble_summary = pd.DataFrame(ensemble_rows).sort_values("group_oof_ipcw", ascending=False)
display(ensemble_summary[["candidate", "group_oof_ipcw", "weights"]])

selected_candidate = ensemble_summary.iloc[0]
selected_model_names = list(selected_candidate["weights"].keys())
selected_weights = selected_candidate["weights"]

print("Selected candidate:", selected_candidate["candidate"])
print("Selected model names:", selected_model_names)
print("Selected weights:", selected_weights)

groupfold_test_ranks = {
    model_name: group_results[model_name]["test_rank"]
    for model_name in selected_model_names
}


# %% [markdown]
# # Step 18: Final full-train predictions with safety guard
#
# `submission_03` uses a fixed 70/30 rule, but only for selected models whose
# final full-train refit succeeds. Failed final models are skipped and weights
# are renormalized.
#
# `submission_04` uses only GroupKFold fold-average test ranks and is unaffected
# by full-train refit failures.

# %%
valid_final_models = {}
skipped_final_models = []

for model_name in selected_model_names:
    try:
        print(f"\nFinal full-train fit for {model_name}...")
        model = clone(models[model_name])

        pp = FoldPreprocessor().fit(X_train_raw)
        X_full_pp = pp.transform(X_train_raw)
        X_test_pp = pp.transform(X_test_raw)

        model.fit(X_full_pp, y)
        pred_full = np.asarray(model.predict(X_test_pp), dtype=float)
        if pred_full.ndim != 1:
            pred_full = pred_full.reshape(-1)

        if not np.isfinite(pred_full).all():
            raise ValueError("non-finite full-train test predictions")

        valid_final_models[model_name] = rank01(pred_full)
        print(f"  {model_name} full-train prediction succeeded.")

    except Exception as exc:
        print(f"Skipping {model_name} from full-train blend: {repr(exc)}")
        skipped_final_models.append(model_name)
        continue
    finally:
        gc.collect()

assert len(valid_final_models) > 0, "No selected models produced valid full-train predictions."

renorm_names = [name for name in selected_model_names if name in valid_final_models]
renorm_weights = np.array([selected_weights[name] for name in renorm_names], dtype=float)
renorm_weights = renorm_weights / renorm_weights.sum()

print("Models used in submission_03 full-train blend:", renorm_names)
print("Renormalized weights:", dict(zip(renorm_names, renorm_weights)))
if skipped_final_models:
    print("Skipped from submission_03 full-train blend:", skipped_final_models)


# Fixed backup: 100% GroupKFold fold-average test ranks.
final_test_risk_groupfold_only = np.zeros(len(X_test_raw), dtype=float)
for model_name in selected_model_names:
    final_test_risk_groupfold_only += selected_weights[model_name] * groupfold_test_ranks[model_name]


# Fixed main: 70% GroupKFold fold-average ranks + 30% successful full-train ranks.
final_test_risk_70_30 = np.zeros(len(X_test_raw), dtype=float)
for model_name, weight in zip(renorm_names, renorm_weights):
    blended_model_rank = (
        0.70 * groupfold_test_ranks[model_name]
        + 0.30 * valid_final_models[model_name]
    )
    final_test_risk_70_30 += weight * blended_model_rank

assert np.isfinite(final_test_risk_groupfold_only).all()
assert np.isfinite(final_test_risk_70_30).all()


# %% [markdown]
# # Step 19: Save submissions

# %%
best_single_rank = group_results[best_single_model]["test_rank"]

top3_equal_weights = normalize_weight_dict({name: 1.0 for name in top3_models})
top3_equal_rank = np.zeros(len(X_test_raw), dtype=float)
for model_name, weight in top3_equal_weights.items():
    top3_equal_rank += weight * group_results[model_name]["test_rank"]

submission_01 = pd.DataFrame({
    "ID": random_submission["ID"].values,
    "risk_score": best_single_rank,
})
submission_02 = pd.DataFrame({
    "ID": random_submission["ID"].values,
    "risk_score": top3_equal_rank,
})
submission_03 = pd.DataFrame({
    "ID": random_submission["ID"].values,
    "risk_score": final_test_risk_70_30,
})
submission_04 = pd.DataFrame({
    "ID": random_submission["ID"].values,
    "risk_score": final_test_risk_groupfold_only,
})

for sub in [submission_01, submission_02, submission_03, submission_04]:
    validate_submission_frame(sub)

submission_paths = {
    "submission_01_best_single_groupcv.csv": submission_01,
    "submission_02_top3_rank_ensemble.csv": submission_02,
    "submission_03_final_group_weighted_rank_ensemble.csv": submission_03,
    "submission_04_groupfold_only_rank_ensemble.csv": submission_04,
}

for filename, sub in submission_paths.items():
    path = OUTPUT_DIR / filename
    sub.to_csv(path, index=False)
    print("Saved:", path)

print("\nFirst submission recommendation:")
print(OUTPUT_DIR / "submission_03_final_group_weighted_rank_ensemble.csv")
print("\nBackup conservative unseen-center submission:")
print(OUTPUT_DIR / "submission_04_groupfold_only_rank_ensemble.csv")


# %% [markdown]
# # Step 20: Submission form fields
#
# Submit first:
# `submission_03_final_group_weighted_rank_ensemble.csv`
#
# Method:
# `Leakage-safe survival rank ensemble using clinical, cytogenetic, and molecular features; GroupKFold by CENTER; CoxPH/Coxnet/GBSA/RSF/ExtraSurvivalTrees when available.`
#
# Parameters:
# `tau=7 IPCW validation; CENTER excluded from model features; fold-local preprocessing; all-missing removal before train-fold imputation; duplicate removal fold-local after train-fold imputation only; molecular feature selection from training data only; Random KFold, LOCO, and prior submissions diagnostics only; fixed 70/30 fold-average/full-train test-rank blend for submission_03; submission_04 uses 100% GroupKFold fold-average test ranks.`
#
# Y test csv file:
# `submission_03_final_group_weighted_rank_ensemble.csv`
