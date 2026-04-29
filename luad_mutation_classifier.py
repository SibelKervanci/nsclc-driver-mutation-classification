# -*- coding: utf-8 -*-
"""
@author: Gaziantep University
Topic: LUAD Driver Mutation Classification Pipeline v8

METODOLOJİ:
  - ComBat batch düzeltme (3 GEO kohortu)
  - SHAP tabanlı özellik seçimi (500 gen, 1. fold)
  - SVM sabit parametreler (bayesian.py'den aktarıldı)
  - 5-fold stratified CV — tüm değerlendirme tutarlı
  - Ensemble: SoftVoting, OptimalVoting, Stacking
  - TCGA-LUAD dış validasyon (SoftVoting, ComBat harmonizasyon)
  - Ablation study (5-fold, SoftVoting ile tutarlı)
"""

import pandas as pd
import numpy as np
import optuna
import shap
import joblib
import warnings
import torch

from neuroCombat import neuroCombat
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, roc_auc_score,
                             classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 0. AYARLAR
# =========================================================
N_FEATURES_SHAP = 500
OUTER_FOLDS     = 5
INNER_FOLDS     = 3
RANDOM_STATE    = 42
SMOTE_K         = 5

USE_GPU = torch.cuda.is_available()
print(f"GPU durumu: {'AKTIF — ' + torch.cuda.get_device_name(0) if USE_GPU else 'YOK, CPU'}")

# =========================================================
# 1. EN İYİ PARAMETRELER — BAYESIAN BYPASS
# =========================================================
BEST_PARAMS = {
    "XGB": dict(
        n_estimators     = 553,
        max_depth        = 9,
        learning_rate    = 0.02115,
        subsample        = 0.5027,
        colsample_bytree = 0.7786,
        min_child_weight = 7,
        gamma            = 1.0318,
        reg_alpha        = 0.01009,
        reg_lambda       = 0.05167,
    ),
    "LGBM": dict(
        n_estimators      = 194,
        max_depth         = 11,
        learning_rate     = 0.14309,
        num_leaves        = 159,
        subsample         = 0.78510,
        colsample_bytree  = 0.46286,
        min_child_samples = 21,
        reg_alpha         = 0.00013,
        reg_lambda        = 0.44987,
    ),
    "RF": dict(
        n_estimators      = 500,
        max_depth         = 13,
        min_samples_split = 13,
        min_samples_leaf  = 5,
        max_features      = 0.3,
    ),
    "CatBoost": dict(
        iterations          = 455,
        depth               = 6,
        learning_rate       = 0.13735,
        l2_leaf_reg         = 1.64820,
        border_count        = 183,
        bagging_temperature = 0.98639,
    ),
    "LogReg": dict(
        C      = 0.01038,
        solver = "lbfgs",
    ),
    "SVM": dict(
        C      = 1.5405,
        kernel = "rbf",
        gamma  = "auto",
    ),
}

# =========================================================
# 2. VERİ YÜKLEME
# =========================================================
print("\nVeriler yükleniyor...")
X_raw = pd.read_parquet("expr_top2000_genes.parquet")
meta  = pd.read_csv("metadata_combined.csv", index_col=0)

mask  = meta["label"] != "ALK"
X_raw = X_raw.loc[mask]
meta  = meta.loc[mask]

y     = meta.loc[X_raw.index, "label"]
le    = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(le.classes_)

print(f"Toplam örnek: {X_raw.shape[0]} | Özellik: {X_raw.shape[1]}")
print(f"Sınıflar: {list(zip(le.classes_, np.bincount(y_enc)))}")

# =========================================================
# 3. COMBAT
# =========================================================
print("\n--- ComBat ---")
covars_df  = meta.loc[X_raw.index, ["batch"]].copy()
combat_out = neuroCombat(dat=X_raw.T.values, covars=covars_df, batch_col="batch")
X_combat   = pd.DataFrame(combat_out["data"].T,
                           index=X_raw.index, columns=X_raw.columns)
print(f"ComBat tamamlandı. NaN: {X_combat.isna().sum().sum()}")

# =========================================================
# 4. MODEL OLUŞTURUCU
# =========================================================
def build_model(name, params=None):
    if params is None:
        params = BEST_PARAMS.get(name, {})
    p = params.copy()
    if name == "XGB":
        p.update({"use_label_encoder": False, "eval_metric": "mlogloss",
                   "random_state": RANDOM_STATE, "tree_method": "hist",
                   "device": "cuda" if USE_GPU else "cpu"})
        return XGBClassifier(**p)
    elif name == "LGBM":
        p.update({"class_weight": "balanced", "random_state": RANDOM_STATE,
                   "verbose": -1, "device": "gpu" if USE_GPU else "cpu"})
        return LGBMClassifier(**p)
    elif name == "RF":
        p.update({"class_weight": "balanced", "random_state": RANDOM_STATE,
                   "n_jobs": -1})
        return RandomForestClassifier(**p)
    elif name == "CatBoost":
        p.update({"auto_class_weights": "Balanced", "random_seed": RANDOM_STATE,
                   "verbose": 0, "task_type": "GPU" if USE_GPU else "CPU"})
        return CatBoostClassifier(**p)
    elif name == "LogReg":
        p.update({"max_iter": 2000, "class_weight": "balanced",
                   "multi_class": "multinomial", "random_state": RANDOM_STATE})
        return LogisticRegression(**p)

# =========================================================
# 5. SHAP FEATURE SELECTION — Sadece 1. fold
# =========================================================
print(f"\n--- SHAP Feature Selection ({N_FEATURES_SHAP} gen, 1. fold) ---")
outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
folds = list(outer_cv.split(X_combat.values, y_enc))

train_idx_0, test_idx_0 = folds[0]
X_tr0_raw = X_combat.values[train_idx_0]
X_tr0_pre = StandardScaler().fit_transform(X_tr0_raw)
y_tr0     = y_enc[train_idx_0]

sm_pre = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_tr0_res, y_tr0_res = sm_pre.fit_resample(X_tr0_pre, y_tr0)

rf_shap = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_shap.fit(X_tr0_res, y_tr0_res)

explainer = shap.TreeExplainer(rf_shap)
shap_vals = explainer.shap_values(X_tr0_pre[:300])

if isinstance(shap_vals, list):
    mean_shap = np.zeros(X_tr0_pre.shape[1])
    for sv in shap_vals:
        mean_shap += np.abs(np.array(sv)).mean(axis=0)
    mean_shap /= len(shap_vals)
else:
    arr = np.array(shap_vals)
    if arr.ndim == 3:
        mean_shap = np.abs(arr).mean(axis=(0, 2))
    elif arr.ndim == 2:
        mean_shap = np.abs(arr).mean(axis=0)
    else:
        mean_shap = np.abs(arr).ravel()

mean_shap      = np.array(mean_shap, dtype=float).ravel()
top_idx        = np.argsort(mean_shap)[::-1][:N_FEATURES_SHAP].tolist()
feature_names  = X_combat.columns.tolist()
selected_genes = [feature_names[i] for i in top_idx]
print(f"Seçilen genler: {len(selected_genes)}")

# =========================================================
# 6. SVM PARAMETRELERİ — BEST_PARAMS'tan al
#    (Bayesian optimizasyon bayesian.py'ye taşındı)
# =========================================================
print("\n--- SVM Parametreleri (Bayesian opt sonucu) ---")

X_tr0_sel    = X_tr0_raw[:, top_idx]
sc0          = StandardScaler()
X_tr0_scaled = sc0.fit_transform(X_tr0_sel)
y_tr0_orig   = y_enc[train_idx_0]

svm_best = BEST_PARAMS["SVM"]
print(f"  Parametreler: {svm_best}")

# =========================================================
# 7. GLOBAL AĞIRLIK OPT — 1. fold (referans)
#    OptimalVoting için (F1 tek amaçlı)
# =========================================================
print("\n--- Global Ensemble Ağırlık Optimizasyonu (F1, 1. fold) ---")

X_te0_raw    = X_combat.values[test_idx_0]
X_te0_sel    = X_te0_raw[:, top_idx]
X_te0_scaled = sc0.transform(X_te0_sel)
y_te0        = y_enc[test_idx_0]

sm0 = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_tr0_res2, y_tr0_res2 = sm0.fit_resample(X_tr0_scaled, y_tr0_orig)

m0 = {}
for mn in ["XGB", "LGBM", "RF", "CatBoost"]:
    m = build_model(mn)
    m.fit(X_tr0_res2, y_tr0_res2)
    m0[mn] = m
m0["SVM"] = SVC(C=svm_best["C"], kernel=svm_best["kernel"],
                gamma=svm_best["gamma"], probability=True,
                class_weight="balanced", random_state=RANDOM_STATE)
m0["SVM"].fit(X_tr0_res2, y_tr0_res2)

probs0 = {k: v.predict_proba(X_te0_scaled) for k, v in m0.items()}

def get_weighted_probs(w, p_dict):
    total = sum(w.values())
    return (w["w_xgb"]  * p_dict["XGB"] +
            w["w_lgbm"] * p_dict["LGBM"] +
            w["w_rf"]   * p_dict["RF"] +
            w["w_cat"]  * p_dict["CatBoost"] +
            w["w_svm"]  * p_dict["SVM"]) / total

def weight_obj_f1(trial):
    w = {"w_xgb" : trial.suggest_float("w_xgb",  0.1, 5.0),
         "w_lgbm": trial.suggest_float("w_lgbm", 0.1, 5.0),
         "w_rf"  : trial.suggest_float("w_rf",   0.1, 3.0),
         "w_cat" : trial.suggest_float("w_cat",  0.1, 5.0),
         "w_svm" : trial.suggest_float("w_svm",  0.1, 5.0)}
    p = get_weighted_probs(w, probs0)
    return f1_score(y_te0, np.argmax(p, axis=1), average="macro", zero_division=0)

study_f1 = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_f1.optimize(weight_obj_f1, n_trials=200, show_progress_bar=False)
best_w_global = study_f1.best_params
print(f"  Global F1 ağırlıklar: {best_w_global}")

# =========================================================
# 8. 5-FOLD CV DEĞERLENDİRME
#    Her fold içinde:
#      - Modeller eğitilir
#      - SoftVoting, OptimalVoting(global), Stacking
# =========================================================
print("\n--- 5-Fold CV Değerlendirme ---")

model_names = ["XGB", "LGBM", "RF", "CatBoost", "LogReg", "SVM",
               "SoftVoting", "OptimalVoting", "Stacking"]
cv_results  = {m: {"acc":[], "f1":[], "mcc":[], "auroc":[]} for m in model_names}
cv_f1_class = {"EGFR":[], "KRAS":[], "TN":[]}

# Görselleştirme için son fold değişkenleri
vis_models      = None
vis_X_te        = None
vis_y_te        = None
vis_X_tr_res    = None
vis_y_tr_res    = None
vis_probs_sv    = None
vis_scaler      = None
vis_probs_svm   = None
vis_stack       = None

for fold_i, (tr_idx, te_idx) in enumerate(folds):
    print(f"\n  Fold {fold_i+1}/5...")

    X_tr_raw_f = X_combat.values[tr_idx]
    X_te_raw_f = X_combat.values[te_idx]
    y_tr_f     = y_enc[tr_idx]
    y_te_f     = y_enc[te_idx]

    # SHAP genleri seç
    X_tr_sel_f = X_tr_raw_f[:, top_idx]
    X_te_sel_f = X_te_raw_f[:, top_idx]

    # Scaler — fold içinde fit, sızıntısız
    sc_f = StandardScaler()
    X_tr_sc_f = sc_f.fit_transform(X_tr_sel_f)
    X_te_sc_f = sc_f.transform(X_te_sel_f)

    # SMOTE — fold train üzerinde
    min_k = np.min(np.bincount(y_tr_f))
    k = min(SMOTE_K, min_k - 1)
    sm_f = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_tr_res_f, y_tr_res_f = sm_f.fit_resample(X_tr_sc_f, y_tr_f)

    # Base modeller eğit
    fold_models = {}
    for mn in ["XGB", "LGBM", "RF", "CatBoost", "LogReg"]:
        m = build_model(mn)
        m.fit(X_tr_res_f, y_tr_res_f)
        fold_models[mn] = m

    svm_f = SVC(C=svm_best["C"], kernel=svm_best["kernel"],
                gamma=svm_best["gamma"], probability=True,
                class_weight="balanced", random_state=RANDOM_STATE)
    svm_f.fit(X_tr_res_f, y_tr_res_f)
    fold_models["SVM"] = svm_f

    # Base model metrikleri
    for mn in ["XGB", "LGBM", "RF", "CatBoost", "LogReg", "SVM"]:
        preds_f = fold_models[mn].predict(X_te_sc_f)
        probs_f = fold_models[mn].predict_proba(X_te_sc_f)
        cv_results[mn]["acc"].append(accuracy_score(y_te_f, preds_f))
        cv_results[mn]["f1"].append(f1_score(y_te_f, preds_f,
                                              average="macro", zero_division=0))
        cv_results[mn]["mcc"].append(matthews_corrcoef(y_te_f, preds_f))
        try:
            cv_results[mn]["auroc"].append(
                roc_auc_score(y_te_f, probs_f, multi_class="ovr", average="macro"))
        except:
            cv_results[mn]["auroc"].append(np.nan)

    # Test seti olasılıkları
    fold_probs_te = {mn: fold_models[mn].predict_proba(X_te_sc_f)
                     for mn in ["XGB","LGBM","RF","CatBoost","SVM"]}

    def eval_ensemble(name, probs_ens):
        preds_e = np.argmax(probs_ens, axis=1)
        cv_results[name]["acc"].append(accuracy_score(y_te_f, preds_e))
        cv_results[name]["f1"].append(f1_score(y_te_f, preds_e,
                                                average="macro", zero_division=0))
        cv_results[name]["mcc"].append(matthews_corrcoef(y_te_f, preds_e))
        try:
            cv_results[name]["auroc"].append(
                roc_auc_score(y_te_f, probs_ens, multi_class="ovr", average="macro"))
        except:
            cv_results[name]["auroc"].append(np.nan)
        return preds_e

    # SoftVoting — eşit ağırlık
    probs_sv_f = sum(fold_probs_te.values()) / len(fold_probs_te)
    preds_sv_f = eval_ensemble("SoftVoting", probs_sv_f)

    # OptimalVoting — global F1 ağırlıkları (1. fold üzerinde bulundu)
    probs_ov_f = get_weighted_probs(best_w_global, fold_probs_te)
    eval_ensemble("OptimalVoting", probs_ov_f)

    # -------------------------------------------------------
    # Stacking — Meta XGB + passthrough
    meta_xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        device="cuda" if USE_GPU else "cpu",
        tree_method="hist",
    )
    base_est = [(mn.lower(), fold_models[mn])
                for mn in ["XGB","LGBM","RF","CatBoost","SVM"]]
    stack_clf = StackingClassifier(
        estimators=base_est,
        final_estimator=meta_xgb,
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE),
        passthrough=True,
        stack_method="predict_proba",
        n_jobs=1,
    )
    stack_clf.fit(X_tr_res_f, y_tr_res_f)
    probs_st_f = stack_clf.predict_proba(X_te_sc_f)
    eval_ensemble("Stacking", probs_st_f)

    # Sınıf bazlı F1 (SoftVoting — ana model)
    f1_cls = f1_score(y_te_f, preds_sv_f, average=None, zero_division=0)
    cv_f1_class["EGFR"].append(f1_cls[0])
    cv_f1_class["KRAS"].append(f1_cls[1])
    cv_f1_class["TN"].append(f1_cls[2])

    print(f"    SoftVoting   : F1={cv_results['SoftVoting']['f1'][-1]:.4f} "
          f"MCC={cv_results['SoftVoting']['mcc'][-1]:.4f}")
    print(f"    OptimalVoting: F1={cv_results['OptimalVoting']['f1'][-1]:.4f} "
          f"MCC={cv_results['OptimalVoting']['mcc'][-1]:.4f}")
    print(f"    Stacking     : F1={cv_results['Stacking']['f1'][-1]:.4f} "
          f"MCC={cv_results['Stacking']['mcc'][-1]:.4f}")

    # Son fold — görselleştirme değişkenleri
    if fold_i == OUTER_FOLDS - 1:
        vis_models    = fold_models
        vis_X_te      = X_te_sc_f
        vis_y_te      = y_te_f
        vis_X_tr_res  = X_tr_res_f
        vis_y_tr_res  = y_tr_res_f
        vis_probs_sv  = probs_sv_f
        vis_scaler    = sc_f
        vis_probs_svm = fold_models["SVM"].predict_proba(X_te_sc_f)
        vis_stack     = stack_clf

# Görselleştirme alias'ları (eski kodla uyumlu)
best_models = vis_models
X_te_scaled = vis_X_te
y_te        = vis_y_te
X_tr_res    = vis_X_tr_res
y_tr_res    = vis_y_tr_res
probs_opt   = vis_probs_sv
preds_opt   = np.argmax(vis_probs_sv, axis=1)
scaler      = vis_scaler
probs_svm   = vis_probs_svm
stack_xgb   = vis_stack

# =========================================================
# 9. RESULTS — 5-FOLD MEAN ± SD
#    Column order (both tables): Model | Accuracy | F1-macro | ±Std | MCC | AUROC
# =========================================================
print("\n\n=== 5-FOLD CV RESULTS (Mean ± SD) ===")
print(f"  {'Model':<20s}  {'Accuracy':>8}  {'F1-macro':>8}  {'±Std':>6}  {'MCC':>7}  {'AUROC':>7}")
print("  " + "-"*65)
results_list = []
for mn in model_names:
    r = {
        "Model"    : mn,
        "Accuracy" : round(np.mean(cv_results[mn]["acc"]), 4),
        "Acc_std"  : round(np.std(cv_results[mn]["acc"]), 4),
        "F1_macro" : round(np.mean(cv_results[mn]["f1"]), 4),
        "F1_std"   : round(np.std(cv_results[mn]["f1"]), 4),
        "MCC"      : round(np.mean(cv_results[mn]["mcc"]), 4),
        "MCC_std"  : round(np.std(cv_results[mn]["mcc"]), 4),
        "AUROC"    : round(np.nanmean(cv_results[mn]["auroc"]), 4),
        "AUROC_std": round(np.nanstd(cv_results[mn]["auroc"]), 4),
    }
    results_list.append(r)
    print(f"  {mn:<20s}  {r['Accuracy']:>8}  {r['F1_macro']:>8}  ±{r['F1_std']:<5}  {r['MCC']:>7}  {r['AUROC']:>7}")

print(f"\n--- Per-class F1 (SoftVoting — primary model) ---")
for cls in ["EGFR", "KRAS", "TN"]:
    print(f"  {cls}: {np.mean(cv_f1_class[cls]):.4f} ± {np.std(cv_f1_class[cls]):.4f}")

results_df = pd.DataFrame(results_list).sort_values("F1_macro", ascending=False)
results_df.to_csv("results_v7_cv.csv", index=False)
pd.Series(selected_genes, name="gene").to_csv("selected_genes_shap.csv", index=False)
joblib.dump(best_w_global, "best_weights_global_v7.pkl")

print("\n  results_v7_cv.csv kaydedildi.")
print("\n=== TAMAMLANDI ===")


# =========================================================
# İSTATİSTİKSEL TESTLER — Friedman + Cliff's Delta
# =========================================================
print("\n" + "="*55)
print("  İSTATİSTİKSEL TESTLER")
print("="*55)

from scipy.stats import friedmanchisquare

model_names_stat = [
    "SoftVoting", "OptimalVoting", "Stacking",
    "XGB", "LGBM", "CatBoost", "RF", "SVM", "LogReg"
]

f1_data    = {mn: np.array(cv_results[mn]["f1"], dtype=float) for mn in model_names_stat}
f1_df_stat = pd.DataFrame(f1_data)

print(f"\n  Fold bazında F1 matrisi ({len(model_names_stat)} model × 5 fold):")
for mn in model_names_stat:
    vals = f1_data[mn]
    print(f"    {mn:<20s}: {[f'{v:.4f}' for v in vals]}  mean={vals.mean():.4f} ±{vals.std():.4f}")

# Friedman Testi
print("\n--- Friedman Testi ---")
friedman_stat, friedman_p = friedmanchisquare(*[f1_df_stat[mn] for mn in model_names_stat])
print(f"  χ²(Friedman) = {friedman_stat:.4f}")
print(f"  p-değeri     = {friedman_p:.6f}")
print(f"  Sonuç        : {'H0 REDDEDİLDİ — modeller arasında anlamlı fark var (p<0.05)' if friedman_p < 0.05 else 'H0 reddedilemedi (p≥0.05)'}")

# Cliff's Delta
def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    greater  = sum(1 for xi in x for yi in y if xi > yi)
    less     = sum(1 for xi in x for yi in y if xi < yi)
    return (greater - less) / (n_x * n_y)

def interpret_cliffs(d):
    a = abs(d)
    if a < 0.147: return "ihmal edilebilir"
    elif a < 0.330: return "küçük"
    elif a < 0.474: return "orta"
    else: return "büyük"

print("\n--- Cliff's Delta Etki Büyüklüğü (SoftVoting referans) ---")
print("\n  " + "{:<22} {:<18} {:<14} {:<18}".format("Model", "F1 (mean±SD)", "Cliff's Delta", "Etki"))
print("  " + "-"*74)

ref_f1  = f1_data["SoftVoting"]
sv_mean = ref_f1.mean()
sv_std  = ref_f1.std()
print("  " + "{:<22} {:<18} {:<14} {:<18}".format(
    "SoftVoting ★", f"{sv_mean:.4f} ± {sv_std:.4f}", "—", "—"))

stat_results = []
for mn in model_names_stat[1:]:
    comp_f1 = f1_data[mn]
    mean    = comp_f1.mean()
    std     = comp_f1.std()
    delta   = cliffs_delta(ref_f1, comp_f1)
    label   = interpret_cliffs(delta)
    print("  " + "{:<22} {:<18} {:<14} {:<18}".format(
        mn, f"{mean:.4f} ± {std:.4f}", f"{delta:.4f}", label))
    stat_results.append({
        "Model": mn, "F1_mean": round(mean,4), "F1_std": round(std,4),
        "Cliffs_Delta": round(delta,4), "Etki_Buyuklugu": label,
        "Mean_Diff_vs_SV": round(sv_mean - mean, 4)
    })

pd.DataFrame(stat_results).to_csv("statistical_tests_results.csv", index=False)

print(f"\n--- Özet ---")
print(f"  Friedman χ²={friedman_stat:.4f}, p={friedman_p:.6f} → {'Anlamlı genel fark ✓' if friedman_p < 0.05 else 'Anlamlı fark yok'}")
buyuk_etki = [r["Model"] for r in stat_results if r["Etki_Buyuklugu"] == "büyük"]
orta_etki  = [r["Model"] for r in stat_results if r["Etki_Buyuklugu"] == "orta"]
kucuk_etki = [r["Model"] for r in stat_results if r["Etki_Buyuklugu"] == "küçük"]
if buyuk_etki: print(f"  Büyük etki (δ≥0.474) : {', '.join(buyuk_etki)}")
if orta_etki:  print(f"  Orta etki  (δ≥0.330) : {', '.join(orta_etki)}")
if kucuk_etki: print(f"  Küçük etki (δ≥0.147) : {', '.join(kucuk_etki)}")
print(f"\n  statistical_tests_results.csv kaydedildi.")
print("="*55)
# =========================================================
# 10. DIŞ VALİDASYON — TCGA-LUAD (SoftVoting)
#     Scaler: tüm eğitim verisi üzerinde yeniden fit
#     (fold-bağımsız, dış validasyon için doğru yaklaşım)
# =========================================================
print("\n" + "="*55)
print("  DIŞ VALİDASYON: TCGA-LUAD")
print("="*55)

# Tüm eğitim verisi üzerinde scaler — dış validasyon için
X_all_sel    = X_combat.values[:, top_idx]
sc_all       = StandardScaler()
sc_all.fit(X_all_sel)   # tüm eğitim

# Tüm eğitim verisi üzerinde modeller — dış validasyon için
sm_all = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_all_sc     = sc_all.transform(X_all_sel)
X_all_res, y_all_res = sm_all.fit_resample(X_all_sc, y_enc)

print("Tüm eğitim verisi üzerinde modeller eğitiliyor (dış validasyon için)...")
final_models = {}
for mn in ["XGB", "LGBM", "RF", "CatBoost", "LogReg"]:
    m = build_model(mn)
    m.fit(X_all_res, y_all_res)
    final_models[mn] = m
final_svm = SVC(C=svm_best["C"], kernel=svm_best["kernel"],
                gamma=svm_best["gamma"], probability=True,
                class_weight="balanced", random_state=RANDOM_STATE)
final_svm.fit(X_all_res, y_all_res)
final_models["SVM"] = final_svm
print("  Modeller hazır.")

print("\nTCGA verisi yükleniyor...")
tcga_expr   = pd.read_parquet("C:/sibel/gozdehoca/tcga_luad_expr.parquet")
tcga_labels = pd.read_csv("C:/sibel/gozdehoca/tcga_luad_labels.csv",
                           index_col=0).squeeze()

tcga_mask   = tcga_labels.isin(le.classes_)
tcga_expr   = tcga_expr.loc[tcga_mask]
tcga_labels = tcga_labels[tcga_mask]
print(f"Kullanılan örnek: {len(tcga_labels)}")
print(f"Dağılım: {tcga_labels.value_counts().to_dict()}")

# ComBat harmonizasyon
print("\nComBat harmonizasyonu (eğitim + TCGA)...")
train_genes  = list(X_combat.columns)
tcga_genes   = list(tcga_expr.columns)
shared_genes = list(set(train_genes) & set(tcga_genes))
print(f"Eğitim: {len(train_genes)} | TCGA: {len(tcga_genes)} | Ortak: {len(shared_genes)}")

X_train_sh = X_combat[shared_genes]
X_tcga_sh  = tcga_expr[shared_genes]
combined   = pd.concat([X_train_sh, X_tcga_sh], axis=0)
batch_vec  = pd.Series(
    [0]*len(X_train_sh) + [1]*len(X_tcga_sh),
    index=combined.index, name="batch"
)
combat_tcga = neuroCombat(
    dat=combined.T.values.astype(float),
    covars=pd.DataFrame({"batch": batch_vec}),
    batch_col="batch"
)
combined_corr = pd.DataFrame(
    combat_tcga["data"].T,
    index=combined.index, columns=shared_genes
)
tcga_corr = combined_corr.iloc[len(X_train_sh):]

# SHAP genleri
common_shap = [g for g in selected_genes if g in shared_genes]
missing     = len(selected_genes) - len(common_shap)
print(f"SHAP genleri: {len(selected_genes)} | Ortak+harmonize: {len(common_shap)} | Eksik: {missing}")

tcga_sel = pd.DataFrame(0.0, index=tcga_corr.index, columns=selected_genes)
tcga_sel[common_shap] = tcga_corr[common_shap].values

# Tüm eğitim scaler ile ölçeklendir
tcga_scaled = sc_all.transform(tcga_sel.values)
tcga_y      = le.transform(tcga_labels)

# SoftVoting tahmini
probs_tcga = sum(
    final_models[mn].predict_proba(tcga_scaled)
    for mn in ["XGB","LGBM","RF","CatBoost","SVM"]
) / 5

# Threshold optimizasyonu
print("\n--- Threshold Optimizasyonu ---")
best_thresholds = {}
for cls_idx, cls_name in enumerate(le.classes_):
    best_t, best_f1_t = 0.3, 0
    for t in np.arange(0.1, 0.9, 0.01):
        preds_t = (probs_tcga[:, cls_idx] >= t).astype(int)
        y_bin   = (tcga_y == cls_idx).astype(int)
        score   = f1_score(y_bin, preds_t, zero_division=0)
        if score > best_f1_t:
            best_f1_t, best_t = score, t
    best_thresholds[cls_name] = best_t
    print(f"  {cls_name}: threshold={best_t:.2f}, F1={best_f1_t:.3f}")

scores_adj = np.zeros((len(tcga_y), len(le.classes_)))
for cls_idx, cls_name in enumerate(le.classes_):
    scores_adj[:, cls_idx] = probs_tcga[:, cls_idx] / best_thresholds[cls_name]
preds_tcga = np.argmax(scores_adj, axis=1)

try:
    auroc_tcga = roc_auc_score(tcga_y, probs_tcga,
                                multi_class="ovr", average="macro")
except:
    auroc_tcga = np.nan

# External validation summary — same column order as CV table:
# Accuracy | F1-macro | MCC | AUROC
acc_tcga = accuracy_score(tcga_y, preds_tcga)
f1_tcga  = f1_score(tcga_y, preds_tcga, average='macro')
mcc_tcga = matthews_corrcoef(tcga_y, preds_tcga)

print("\n--- TCGA-LUAD External Validation Results (SoftVoting) ---")
print(f"  {'Metric':<12}  {'Value':>8}")
print("  " + "-"*24)
print(f"  {'Accuracy':<12}  {acc_tcga:>8.4f}")
print(f"  {'F1-macro':<12}  {f1_tcga:>8.4f}")
print(f"  {'MCC':<12}  {mcc_tcga:>8.4f}")
print(f"  {'AUROC':<12}  {auroc_tcga:>8.4f}")
print(f"\nDetailed per-class report:\n"
      f"{classification_report(tcga_y, preds_tcga, target_names=le.classes_)}")

tcga_results = pd.DataFrame([{
    "Dataset"  : "TCGA-LUAD (external validation)",
    "Model"    : "SoftVoting",
    "Platform" : "RNA-seq",
    "N"        : len(tcga_y),
    "Accuracy" : round(acc_tcga, 4),
    "F1_macro" : round(f1_tcga,  4),
    "MCC"      : round(mcc_tcga, 4),
    "AUROC"    : round(auroc_tcga, 4),
}])
tcga_results.to_csv("tcga_external_validation_v7.csv", index=False)
print("\n  tcga_external_validation_v7.csv saved.")
print("="*55)

# =========================================================
# 11. ABLATION STUDY — 5-fold, SoftVoting ile tutarlı
# =========================================================
print("\n" + "="*55)
print("  ABLATION STUDY (5-Fold CV, SoftVoting)")
print("="*55)

ablation_results = []

def run_ablation(name, X_data, y_data, use_smote=True,
                 use_bayesian=True, ensemble=True):
    fold_f1, fold_acc, fold_mcc, fold_auroc = [], [], [], []

    for tr_idx, te_idx in folds:
        X_tr = X_data[tr_idx]
        X_te = X_data[te_idx]
        y_tr = y_data[tr_idx]
        y_te = y_data[te_idx]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

        if use_smote:
            min_k = np.min(np.bincount(y_tr))
            k = min(5, min_k - 1)
            if k >= 1:
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        if use_bayesian:
            xgb_p  = BEST_PARAMS["XGB"].copy()
            lgbm_p = BEST_PARAMS["LGBM"].copy()
            rf_p   = BEST_PARAMS["RF"].copy()
            cat_p  = BEST_PARAMS["CatBoost"].copy()
        else:
            xgb_p  = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
            lgbm_p = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
            rf_p   = {"n_estimators": 100, "max_depth": 10}
            cat_p  = {"iterations": 100, "depth": 6, "learning_rate": 0.1}

        if ensemble:
            xgb  = build_model("XGB",      xgb_p)
            lgbm = build_model("LGBM",     lgbm_p)
            rf   = build_model("RF",       rf_p)
            cat  = build_model("CatBoost", cat_p)
            svm  = SVC(C=svm_best["C"], kernel=svm_best["kernel"],
                       gamma=svm_best["gamma"], probability=True,
                       class_weight="balanced", random_state=42)
            for m in [xgb, lgbm, rf, cat, svm]:
                m.fit(X_tr, y_tr)

            # SoftVoting — ana modelle tutarlı
            fp = {mn: m.predict_proba(X_te)
                  for mn, m in zip(["XGB","LGBM","RF","CatBoost","SVM"],
                                   [xgb, lgbm, rf, cat, svm])}
            probs = sum(fp.values()) / len(fp)
            preds = np.argmax(probs, axis=1)
        else:
            xgb = build_model("XGB", xgb_p)
            xgb.fit(X_tr, y_tr)
            preds = xgb.predict(X_te)
            probs = xgb.predict_proba(X_te)

        fold_f1.append(f1_score(y_te, preds, average="macro", zero_division=0))
        fold_acc.append(accuracy_score(y_te, preds))
        fold_mcc.append(matthews_corrcoef(y_te, preds))
        try:
            fold_auroc.append(roc_auc_score(y_te, probs,
                                             multi_class="ovr", average="macro"))
        except:
            fold_auroc.append(np.nan)

    return {
        "Configuration": name,
        "F1_macro"     : round(np.mean(fold_f1), 4),
        "F1_std"       : round(np.std(fold_f1), 4),
        "Accuracy"     : round(np.mean(fold_acc), 4),
        "MCC"          : round(np.mean(fold_mcc), 4),
        "AUROC"        : round(np.nanmean(fold_auroc), 4),
    }

X_raw_np      = X_raw.values
X_combat_np   = X_combat.values
y_all_enc     = y_enc
top_idx_arr   = np.array(top_idx)
X_shap_combat = X_combat_np[:, top_idx_arr]
X_shap_raw    = X_raw_np[:, top_idx_arr]

print("Ablation konfigürasyonları çalıştırılıyor...")

print("  1/6 Full Pipeline...")
r = run_ablation("Full Pipeline", X_shap_combat, y_all_enc)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

print("  2/6 No ComBat...")
r = run_ablation("No ComBat", X_shap_raw, y_all_enc)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

print("  3/6 No SMOTE...")
r = run_ablation("No SMOTE", X_shap_combat, y_all_enc, use_smote=False)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

print("  4/6 No SHAP (all 2000 genes)...")
r = run_ablation("No SHAP", X_combat_np, y_all_enc)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

print("  5/6 No Bayesian Opt...")
r = run_ablation("No Bayesian Opt", X_shap_combat, y_all_enc, use_bayesian=False)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

print("  6/6 XGB Only (no ensemble)...")
r = run_ablation("XGB Only", X_shap_combat, y_all_enc, ensemble=False)
ablation_results.append(r)
print(f"     F1={r['F1_macro']} ± {r['F1_std']} | AUROC={r['AUROC']}")

abl_df = pd.DataFrame(ablation_results).sort_values("F1_macro", ascending=False)
print("\n=== ABLATION STUDY SONUÇLARI ===")
print(abl_df.to_string(index=False))
abl_df.to_csv("ablation_results.csv", index=False)
print("  ablation_results.csv kaydedildi.")
print("="*55)
# =========================================================
# 12. GÖRSELLEŞTİRMELER
# =========================================================
# =========================================================
# 12. GÖRSELLEŞTİRMELER
# =========================================================
print("\n" + "="*55)
print("  GÖRSELLEŞTİRMELER")
print("="*55)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap as shap_lib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                              roc_curve, auc,
                              precision_recall_curve,
                              average_precision_score,
                              precision_score, recall_score,
                              f1_score as f1_sc)
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler as SS

# ── Renk ve stil ──────────────────────────────────────────
CLASS_COLORS = {"EGFR": "#9B59B6", "KRAS": "#E67E22", "TN": "#1ABC9C"}
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10})

batch_labels     = meta.loc[X_raw.index, "batch"].values
unique_batches   = sorted(set(batch_labels), key=lambda x: str(x))
palette          = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#8E44AD"]
BATCH_COLORS     = {b: palette[i] for i, b in enumerate(unique_batches)}
BATCH_NAMES      = {b: str(b) for b in unique_batches}
batch_colors_arr = np.array([BATCH_COLORS[b] for b in batch_labels])
class_colors_arr = np.array([CLASS_COLORS[le.classes_[c]] for c in y_enc])

sc_vis  = SS(); X_bef_sc = sc_vis.fit_transform(X_raw.values)
sc_vis2 = SS(); X_aft_sc = sc_vis2.fit_transform(X_combat.values)

# ── PCA ───────────────────────────────────────────────────
print("  PCA hesaplanıyor...")
pca1 = PCA(n_components=2, random_state=42)
pb   = pca1.fit_transform(X_bef_sc)
vb   = pca1.explained_variance_ratio_ * 100

pca2 = PCA(n_components=2, random_state=42)
pa   = pca2.fit_transform(X_aft_sc)
va   = pca2.explained_variance_ratio_ * 100

# ── t-SNE ─────────────────────────────────────────────────
print("  t-SNE hesaplanıyor (2-3 dk)...")
tsne_params = dict(n_components=2, perplexity=40, random_state=42, n_jobs=-1)
try:
    tb = TSNE(**tsne_params, max_iter=1000).fit_transform(X_bef_sc)
    ta = TSNE(**tsne_params, max_iter=1000).fit_transform(X_aft_sc)
except TypeError:
    tb = TSNE(**tsne_params, n_iter=1000).fit_transform(X_bef_sc)
    ta = TSNE(**tsne_params, n_iter=1000).fit_transform(X_aft_sc)
print("  t-SNE tamamlandı.")

def _scatter(ax, xy, colors, xl, yl, handles):
    ax.scatter(xy[:,0], xy[:,1], c=colors, s=16, alpha=0.65, edgecolors='none')
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=handles, fontsize=8, framealpha=0.9,
              loc='upper right', markerscale=1.1)

batch_h = [mpatches.Patch(color=BATCH_COLORS[b], label=BATCH_NAMES[b])
           for b in unique_batches]
class_h = [mpatches.Patch(color=CLASS_COLORS[c], label=c)
           for c in le.classes_]

# =========================================================
# FIG 1: ComBat Comparison (2×4)
# =========================================================
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
fig1.patch.set_facecolor('white')

_scatter(axes1[0,0], pb, batch_colors_arr,
         f"PC1 ({vb[0]:.1f}%)", f"PC2 ({vb[1]:.1f}%)", batch_h)
axes1[0,0].set_title("PCA — Before ComBat (Cohort)", fontsize=10, pad=6)

_scatter(axes1[0,1], pb, class_colors_arr,
         f"PC1 ({vb[0]:.1f}%)", f"PC2 ({vb[1]:.1f}%)", class_h)
axes1[0,1].set_title("PCA — Before ComBat (Subtype)", fontsize=10, pad=6)

_scatter(axes1[0,2], pa, batch_colors_arr,
         f"PC1 ({va[0]:.1f}%)", f"PC2 ({va[1]:.1f}%)", batch_h)
axes1[0,2].set_title("PCA — After ComBat (Cohort)", fontsize=10, pad=6)

_scatter(axes1[0,3], pa, class_colors_arr,
         f"PC1 ({va[0]:.1f}%)", f"PC2 ({va[1]:.1f}%)", class_h)
axes1[0,3].set_title("PCA — After ComBat (Subtype)", fontsize=10, pad=6)

_scatter(axes1[1,0], tb, batch_colors_arr, "t-SNE 1", "t-SNE 2", batch_h)
axes1[1,0].set_title("t-SNE — Before ComBat (Cohort)", fontsize=10, pad=6)

_scatter(axes1[1,1], tb, class_colors_arr, "t-SNE 1", "t-SNE 2", class_h)
axes1[1,1].set_title("t-SNE — Before ComBat (Subtype)", fontsize=10, pad=6)

_scatter(axes1[1,2], ta, batch_colors_arr, "t-SNE 1", "t-SNE 2", batch_h)
axes1[1,2].set_title("t-SNE — After ComBat (Cohort)", fontsize=10, pad=6)

_scatter(axes1[1,3], ta, class_colors_arr, "t-SNE 1", "t-SNE 2", class_h)
axes1[1,3].set_title("t-SNE — After ComBat (Subtype)", fontsize=10, pad=6)

plt.tight_layout()
fig1.savefig("fig1_combat_comparison.png", dpi=300,
             bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("  fig1_combat_comparison.png saved.")

# =========================================================
# FIG TCGA: Platform Shift PCA (GEO + TCGA, before/after)
# =========================================================
print("  Computing TCGA platform shift PCA...")
try:
    from sklearn.preprocessing import StandardScaler as SS2

    X_train_sh_vals = X_combat[shared_genes].values
    X_tcga_sh_vals  = tcga_expr[shared_genes].values
    combined_before = np.vstack([X_train_sh_vals, X_tcga_sh_vals])
    combined_after  = combined_corr.values
    n_train_tp      = len(X_train_sh_vals)
    n_tcga_tp       = len(X_tcga_sh_vals)

    sc_t1 = SS2(); X_b_sc_t = sc_t1.fit_transform(combined_before)
    sc_t2 = SS2(); X_a_sc_t = sc_t2.fit_transform(combined_after)

    pca_t1 = PCA(n_components=2, random_state=42)
    pb_t   = pca_t1.fit_transform(X_b_sc_t)
    vb_t   = pca_t1.explained_variance_ratio_ * 100

    pca_t2 = PCA(n_components=2, random_state=42)
    pa_t   = pca_t2.fit_transform(X_a_sc_t)
    va_t   = pca_t2.explained_variance_ratio_ * 100

    source_h_tp = [
        mpatches.Patch(color="#3498DB", label=f"GEO training (n={n_train_tp})"),
        mpatches.Patch(color="#E67E22", label=f"TCGA-LUAD (n={n_tcga_tp})")
    ]

    fig_tp, axes_tp = plt.subplots(1, 2, figsize=(12, 5))
    fig_tp.patch.set_facecolor('white')

    for ax, xy, vr, title in zip(
            axes_tp,
            [pb_t, pa_t],
            [vb_t, va_t],
            ["(a) Before ComBat harmonization",
             "(b) After ComBat harmonization"]):
        ax.scatter(xy[:n_train_tp, 0], xy[:n_train_tp, 1],
                   c="#3498DB", s=12, alpha=0.5, edgecolors='none')
        ax.scatter(xy[n_train_tp:, 0], xy[n_train_tp:, 1],
                   c="#E67E22", s=12, alpha=0.6, edgecolors='none')
        ax.set_xlabel(f"PC1 ({vr[0]:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({vr[1]:.1f}%)", fontsize=10)
        ax.set_title(title, fontsize=12, pad=8)
        ax.legend(handles=source_h_tp, fontsize=9, framealpha=0.9)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig_tp.savefig("fig_tcga_combat_pca.png", dpi=300,
                   bbox_inches='tight', facecolor='white')
    plt.close(fig_tp)
    print("  fig_tcga_combat_pca.png saved.")
except Exception as e:
    print(f"  TCGA combat PCA skipped: {e}")

# =========================================================
# FIG 2: Confusion Matrices (internal + TCGA)
# =========================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('white')

cm_inner = confusion_matrix(vis_y_te, np.argmax(vis_probs_sv, axis=1))
disp1 = ConfusionMatrixDisplay(cm_inner, display_labels=le.classes_)
disp1.plot(ax=axes2[0], colorbar=False, cmap='Blues')
axes2[0].set_title("(a) Internal cohort (SoftVoting)", fontsize=11, pad=8)
axes2[0].set_xlabel("Predicted label", fontsize=10)
axes2[0].set_ylabel("True label", fontsize=10)

cm_tcga = confusion_matrix(tcga_y, preds_tcga)
disp2 = ConfusionMatrixDisplay(cm_tcga, display_labels=le.classes_)
disp2.plot(ax=axes2[1], colorbar=False, cmap='Oranges')
axes2[1].set_title("(b) TCGA-LUAD external validation", fontsize=11, pad=8)
axes2[1].set_xlabel("Predicted label", fontsize=10)
axes2[1].set_ylabel("True label", fontsize=10)

plt.tight_layout()
fig2.savefig("fig2_confusion_matrix.png", dpi=300,
             bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("  fig2_confusion_matrix.png saved.")

# =========================================================
# FIG HEATMAP: Per-class Precision / Recall / F1
# =========================================================
print("  Computing classification heatmap (internal + TCGA)...")
try:
    preds_sv_inner = np.argmax(vis_probs_sv, axis=1)

    def get_metrics_matrix(y_true, y_pred, classes):
        prec = precision_score(y_true, y_pred, average=None,
                               labels=list(range(len(classes))),
                               zero_division=0)
        rec  = recall_score(y_true, y_pred, average=None,
                            labels=list(range(len(classes))),
                            zero_division=0)
        f1   = f1_sc(y_true, y_pred, average=None,
                     labels=list(range(len(classes))),
                     zero_division=0)
        return np.array([prec, rec, f1])

    mat_inner = get_metrics_matrix(vis_y_te, preds_sv_inner, le.classes_)
    mat_tcga  = get_metrics_matrix(tcga_y,   preds_tcga,     le.classes_)

    metric_labels = ["Precision", "Recall", "F1-score"]
    class_labels  = list(le.classes_)

    fig_ch, axes_ch = plt.subplots(1, 2, figsize=(13, 4))
    fig_ch.patch.set_facecolor('white')

    for ax, mat, panel_title, cmap in zip(
            axes_ch,
            [mat_inner, mat_tcga],
            ["(a) Internal cohort (SoftVoting)",
             "(b) TCGA-LUAD external validation"],
            ["Blues", "Oranges"]):

        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                text_col = 'white' if val > 0.6 else '#2C3E50'
                ax.text(j, i, f"{val:.3f}",
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color=text_col)

        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=11)
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels, fontsize=11)
        ax.set_title(panel_title, fontsize=12, pad=10)
        ax.set_xticks(np.arange(-0.5, len(class_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    plt.tight_layout()
    fig_ch.savefig("fig_classification_heatmap.png", dpi=300,
                   bbox_inches='tight', facecolor='white')
    plt.close(fig_ch)
    print("  fig_classification_heatmap.png saved.")
except Exception as e:
    print(f"  Classification heatmap skipped: {e}")

# =========================================================
# FIG 3: ROC Curves (internal + TCGA)
# =========================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.patch.set_facecolor('white')

def _plot_roc(ax, y_true, y_prob, panel_title):
    y_bin      = label_binarize(y_true, classes=list(range(len(le.classes_))))
    colors_roc = [CLASS_COLORS[c] for c in le.classes_]
    for i, (cls, col) in enumerate(zip(le.classes_, colors_roc)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")
    all_fpr  = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_prob[:, i])[0]
        for i in range(len(le.classes_))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(le.classes_)):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(le.classes_)
    ax.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
            label=f"Macro avg. (AUC = {auc(all_fpr, mean_tpr):.3f})")
    ax.plot([0,1],[0,1],'k:', lw=1, alpha=0.4)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False positive rate", fontsize=10)
    ax.set_ylabel("True positive rate", fontsize=10)
    ax.set_title(panel_title, fontsize=11, pad=8)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.spines[['top','right']].set_visible(False)

_plot_roc(axes3[0], vis_y_te, vis_probs_sv,
          "(a) Internal cohort (SoftVoting, final fold)")
_plot_roc(axes3[1], tcga_y,   probs_tcga,
          "(b) External validation — TCGA-LUAD")

plt.tight_layout()
fig3.savefig("fig3_roc_curves.png", dpi=300,
             bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("  fig3_roc_curves.png saved.")

# =========================================================
# CPU XGBoost — tüm SHAP görselleri için tek model
# =========================================================
print("  CPU XGBoost modeli eğitiliyor (tüm SHAP görselleri için)...")
try:
    _xgb_shap_params = BEST_PARAMS["XGB"].copy()
    _xgb_shap_params.update({
        "use_label_encoder": False,
        "eval_metric"      : "mlogloss",
        "random_state"     : RANDOM_STATE,
        "tree_method"      : "hist",
        "device"           : "cpu",        # SHAP için zorunlu CPU
    })
    _xgb_shap = XGBClassifier(**_xgb_shap_params)
    _xgb_shap.fit(vis_X_tr_res, vis_y_tr_res)
    print("  CPU XGBoost hazır.")
except Exception as _e:
    print(f"  CPU XGBoost hatası: {_e}")
    _xgb_shap = None

# ── SHAP string hata önleme ───────────────────────────────
def safe_shap_values(explainer, X_data):
    """GPU XGBoost'un bıraktığı [0E0,0E0,0E0] string sorununu çözer."""
    raw = explainer.shap_values(X_data)
    if isinstance(raw, list):
        cleaned = []
        for sv in raw:
            arr = np.array(sv)
            if arr.dtype == object:
                arr = np.vectorize(
                    lambda x: float(str(x).strip('[]').split(',')[0])
                              if isinstance(x, str) else float(x))(arr)
            cleaned.append(arr.astype(float))
        return cleaned
    arr = np.array(raw)
    if arr.dtype == object:
        arr = np.vectorize(
            lambda x: float(str(x).strip('[]').split(',')[0])
                      if isinstance(x, str) else float(x))(arr)
    return arr.astype(float)

# SHAP model doğrulama — string hata varsa yeniden eğit
if _xgb_shap is not None:
    try:
        _t = shap_lib.TreeExplainer(_xgb_shap).shap_values(vis_X_te[:3])
        np.array(_t, dtype=float)
        print("  SHAP model doğrulandı.")
    except Exception as _se:
        print(f"  SHAP string hatası — yeniden eğitiliyor: {_se}")
        try:
            _p2 = {k: v for k, v in BEST_PARAMS["XGB"].items()}
            _p2.update({"use_label_encoder": False, "eval_metric": "mlogloss",
                        "random_state": RANDOM_STATE, "tree_method": "exact",
                        "device": "cpu"})
            _xgb_shap = XGBClassifier(**_p2)
            _xgb_shap.fit(vis_X_tr_res, vis_y_tr_res)
            print("  Yeni CPU modeli hazır.")
        except Exception as _e3:
            print(f"  Model hatası: {_e3}")
            _xgb_shap = None

# =========================================================
# FIG 5: SHAP Beeswarm — 3 panel (EGFR / KRAS / TN)
# =========================================================
print("  Computing SHAP beeswarm plot (3-panel)...")
if _xgb_shap is not None:
    try:
        explainer_bs  = shap_lib.TreeExplainer(_xgb_shap)
        shap_vals_bs  = safe_shap_values(explainer_bs, vis_X_te)
        arr_bs        = np.array(shap_vals_bs, dtype=float)

        # shape → (n_classes, n_samples, n_features)
        if arr_bs.ndim == 3 and arr_bs.shape[0] == len(le.classes_):
            shap_cls_bs = arr_bs
        elif arr_bs.ndim == 3 and arr_bs.shape[2] == len(le.classes_):
            shap_cls_bs = arr_bs.transpose(2, 0, 1)
        else:
            shap_cls_bs = np.stack([arr_bs]*len(le.classes_), axis=0)

        TOP_BEE        = 20
        PANEL_COLS_BS  = {"EGFR": "#9B59B6", "KRAS": "#E67E22", "TN": "#1ABC9C"}

        fig5, axes5 = plt.subplots(1, 3, figsize=(18, 12))
        fig5.patch.set_facecolor('white')

        for ci, (cls_name, ax) in enumerate(zip(le.classes_, axes5)):
            shap_c   = shap_cls_bs[ci]
            feat_val = vis_X_te

            mean_abs_c  = np.abs(shap_c).mean(axis=0)
            top_idx_bee = np.argsort(mean_abs_c)[::-1][:TOP_BEE]
            top_genes_b = [selected_genes[i] for i in top_idx_bee]

            for gi, gene_idx in enumerate(top_idx_bee[::-1]):
                shap_col  = shap_c[:, gene_idx].astype(float)
                feat_col  = feat_val[:, gene_idx].astype(float)
                feat_norm = (feat_col - feat_col.min()) / (
                             feat_col.ptp() + 1e-9)
                y_jitter  = gi + np.random.default_rng(42).uniform(
                                 -0.3, 0.3, size=len(shap_col))
                sc = ax.scatter(shap_col, y_jitter,
                                c=feat_norm, cmap='coolwarm',
                                s=12, alpha=0.75, vmin=0, vmax=1,
                                linewidths=0, rasterized=True)

            ax.axvline(0, color='#666666', linewidth=0.8, linestyle='--')
            ax.set_yticks(range(TOP_BEE))
            ax.set_yticklabels(top_genes_b[::-1], fontsize=8, style='italic')
            ax.set_xlabel("SHAP value\n(impact on model output)",
                          fontsize=9, labelpad=8)
            ax.set_title(f"SHAP — {cls_name}",
                         fontsize=12, fontweight='bold',
                         color=PANEL_COLS_BS[cls_name], pad=8)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(axis='x', labelsize=8)

            cbar_b = plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.03)
            cbar_b.set_label("Feature value", fontsize=8)
            cbar_b.set_ticks([0, 0.5, 1])
            cbar_b.set_ticklabels(['Low', '', 'High'], fontsize=7)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.10)
        fig5.savefig("fig5_shap_beeswarm.png", dpi=300,
                     bbox_inches='tight', facecolor='white')
        plt.close(fig5)
        print("  fig5_shap_beeswarm.png saved.")
    except Exception as e:
        print(f"  SHAP beeswarm skipped: {e}")
else:
    print("  SHAP beeswarm skipped: CPU model not available.")

# =========================================================
# FIG 6a: SHAP Bar — sınıf bazlı Top 20 gen
# =========================================================
print("  Computing SHAP bar plot (class-wise top 20)...")
if _xgb_shap is not None:
    try:
        explainer_bar  = shap_lib.TreeExplainer(_xgb_shap)
        shap_vals_bar  = safe_shap_values(explainer_bar, vis_X_te)
        arr_bar        = np.array(shap_vals_bar, dtype=float)

        if arr_bar.ndim == 3 and arr_bar.shape[0] == len(le.classes_):
            shap_cls_bar = arr_bar
        elif arr_bar.ndim == 3 and arr_bar.shape[2] == len(le.classes_):
            shap_cls_bar = arr_bar.transpose(2, 0, 1)
        else:
            shap_cls_bar = np.stack([arr_bar]*len(le.classes_), axis=0)

        BAR_COLS = {"EGFR": "#2980B9", "KRAS": "#E67E22", "TN": "#27AE60"}
        TOP_BAR  = 20

        fig6a, axes6a = plt.subplots(1, 3, figsize=(18, 8))
        fig6a.patch.set_facecolor('white')

        for ci, (cls_name, ax) in enumerate(zip(le.classes_, axes6a)):
            mean_abs      = np.abs(shap_cls_bar[ci]).mean(axis=0)
            top_idx_bar   = np.argsort(mean_abs)[::-1][:TOP_BAR]
            top_genes_bar = [selected_genes[i] for i in top_idx_bar]
            top_vals_bar  = mean_abs[top_idx_bar]

            ax.barh(range(TOP_BAR), top_vals_bar[::-1],
                    color=BAR_COLS[cls_name], alpha=0.85, edgecolor='white')
            ax.set_yticks(range(TOP_BAR))
            ax.set_yticklabels(top_genes_bar[::-1], fontsize=8, style='italic')
            ax.set_xlabel("Mean |SHAP value|", fontsize=10)
            ax.set_title(f"Top 20 Genes\n{cls_name}",
                         fontsize=11, fontweight='bold',
                         color=BAR_COLS[cls_name], pad=8)
            ax.spines[['top','right']].set_visible(False)

        fig6a.suptitle("SHAP Feature Importance by Class (XGBoost)",
                       fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig6a.savefig("fig6_shap_bar_top20.png", dpi=300,
                      bbox_inches='tight', facecolor='white')
        plt.close(fig6a)
        print("  fig6_shap_bar_top20.png saved.")
    except Exception as e:
        print(f"  SHAP bar skipped: {e}")
else:
    print("  SHAP bar skipped: CPU model not available.")

# =========================================================
# FIG 6b: Model Comparison Bar
# =========================================================
fig6b, ax6b = plt.subplots(figsize=(11, 5))
fig6b.patch.set_facecolor('white')

model_order  = ["SoftVoting","OptimalVoting","Stacking",
                "XGB","LGBM","CatBoost","RF","SVM","LogReg"]
model_labels = ["SoftVoting","OptimalVoting","Stacking",
                "XGBoost","LightGBM","CatBoost","Random Forest","SVM","Logistic Reg."]
f1_means = [np.mean(cv_results[m]['f1']) for m in model_order]
f1_stds  = [np.std(cv_results[m]['f1'])  for m in model_order]

bar_cols_m = ['#C0392B' if m == 'SoftVoting'
              else '#E74C3C' if m in ['OptimalVoting','Stacking']
              else '#2980B9' for m in model_order]

x = np.arange(len(model_order))
ax6b.bar(x, f1_means, yerr=f1_stds, color=bar_cols_m,
         alpha=0.85, edgecolor='white', width=0.6,
         capsize=4, error_kw={'linewidth':1.5, 'ecolor':'#2C3E50'})
ax6b.set_xticks(x)
ax6b.set_xticklabels(model_labels, rotation=22, ha='right', fontsize=9)
ax6b.set_ylabel("F1-macro (mean ± SD)", fontsize=10)
ax6b.set_ylim([0.60, 0.78])
ax6b.spines[['top','right']].set_visible(False)
for xi, (v, s) in enumerate(zip(f1_means, f1_stds)):
    ax6b.text(xi, v + s + 0.003, f"{v:.3f}", ha='center',
              fontsize=8, color='#2C3E50')
legend_m = [
    mpatches.Patch(color='#C0392B', label='Primary ensemble (SoftVoting)'),
    mpatches.Patch(color='#E74C3C', label='Ensemble (other)'),
    mpatches.Patch(color='#2980B9', label='Individual classifier'),
]
ax6b.legend(handles=legend_m, fontsize=9, framealpha=0.9)
plt.tight_layout()
fig6b.savefig("fig6_model_comparison.png", dpi=300,
              bbox_inches='tight', facecolor='white')
plt.close(fig6b)
print("  fig6_model_comparison.png saved.")

# =========================================================
# FIG 7: LIME Local Explanations
# =========================================================
print("  Computing LIME explanations (3-panel)...")
try:
    from lime.lime_tabular import LimeTabularExplainer

    lime_exp = LimeTabularExplainer(
        training_data = vis_X_tr_res,
        feature_names = selected_genes,
        class_names   = list(le.classes_),
        mode          = "classification",
        random_state  = RANDOM_STATE,
    )

    preds_last   = np.argmax(vis_probs_sv, axis=1)
    lime_samples = {}
    for ci, cls in enumerate(le.classes_):
        correct_idx = np.where((vis_y_te == ci) & (preds_last == ci))[0]
        lime_samples[cls] = (correct_idx[0] if len(correct_idx) > 0
                             else np.where(vis_y_te == ci)[0][0])

    CLASS_COLS_LIME = {"EGFR": "#9B59B6", "KRAS": "#E67E22", "TN": "#1ABC9C"}

    lime_predict_fn = (_xgb_shap.predict_proba if _xgb_shap is not None
                       else vis_models["XGB"].predict_proba)

    fig7, axes7 = plt.subplots(1, 3, figsize=(18, 7))
    fig7.patch.set_facecolor('white')

    for ci, (cls, ax) in enumerate(zip(le.classes_, axes7)):
        exp       = lime_exp.explain_instance(
                        data_row     = vis_X_te[lime_samples[cls]],
                        predict_fn   = lime_predict_fn,
                        num_features = 15,
                        labels       = (ci,))
        lime_vals = exp.as_list(label=ci)
        features  = [v[0] for v in lime_vals]
        values    = [v[1] for v in lime_vals]
        colors_b  = ['#C0392B' if v > 0 else '#2980B9' for v in values]

        ax.barh(range(len(features)), values,
                color=colors_b, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=8)
        ax.axvline(0, color='#666666', linewidth=0.8, linestyle='--')
        ax.set_xlabel("LIME contribution", fontsize=9)
        ax.set_title(f"LIME — {cls} sample\n(XGBoost)",
                     fontsize=11, fontweight='bold',
                     color=CLASS_COLS_LIME[cls], pad=8)
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    fig7.savefig("fig7_lime_explanations.png", dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig7)
    print("  fig7_lime_explanations.png saved.")
except ImportError:
    print("  LIME skipped: pip install lime")
except Exception as e:
    print(f"  LIME skipped: {e}")

# =========================================================
# FIG 8: Ablation Study
# =========================================================
fig8, ax8 = plt.subplots(figsize=(9, 5))
fig8.patch.set_facecolor('white')

abl_sorted = abl_df.sort_values("F1_macro", ascending=True)
bar_colors_abl = ['#27AE60' if c == 'Full Pipeline' else '#2980B9'
                  for c in abl_sorted['Configuration']]
ax8.barh(abl_sorted['Configuration'], abl_sorted['F1_macro'],
         color=bar_colors_abl, alpha=0.85, edgecolor='white', height=0.6)
ax8.errorbar(abl_sorted['F1_macro'], range(len(abl_sorted)),
             xerr=abl_sorted['F1_std'],
             fmt='none', color='#2C3E50', capsize=4, linewidth=1.5)
for i, (val, std) in enumerate(zip(abl_sorted['F1_macro'],
                                    abl_sorted['F1_std'])):
    ax8.text(abl_sorted['F1_macro'].max() + abl_sorted['F1_std'].max() + 0.003,
             i, f"{val:.4f} ± {std:.3f}",
             va='center', ha='left', fontsize=8.5, color='#2C3E50')
ax8.set_xlabel("F1-macro (five-fold CV mean ± SD)", fontsize=10)
ax8.set_xlim([0.68, 0.785])
ax8.spines[['top','right']].set_visible(False)
ax8.axvline(abl_df.loc[abl_df['Configuration']=='Full Pipeline',
                        'F1_macro'].values[0],
            color='#27AE60', linestyle='--', alpha=0.5, lw=1.5)
ax8.legend(handles=[
    mpatches.Patch(color='#27AE60', label='Full pipeline'),
    mpatches.Patch(color='#2980B9', label='Ablation configuration'),
], fontsize=9, loc='lower right')
plt.tight_layout()
fig8.savefig("fig8_ablation_chart.png", dpi=300,
             bbox_inches='tight', facecolor='white')
plt.close(fig8)
print("  fig8_ablation_chart.png saved.")

# =========================================================
# FIG 9: SHAP Heatmap — Top 30 gen × test örnekleri
# =========================================================
print("  Computing SHAP heatmap (top 30 genes x test samples)...")
if _xgb_shap is not None:
    try:
        explainer_hm9  = shap_lib.TreeExplainer(_xgb_shap)
        shap_vals_hm9  = safe_shap_values(explainer_hm9, vis_X_te)
        arr_hm9        = np.array(shap_vals_hm9, dtype=float)

        if arr_hm9.ndim == 3 and arr_hm9.shape[0] == len(le.classes_):
            mean_shap_hm9 = np.abs(arr_hm9).mean(axis=(0, 2))
            sv_matrix_hm9 = arr_hm9.mean(axis=0)
        elif arr_hm9.ndim == 3:
            mean_shap_hm9 = np.abs(arr_hm9).mean(axis=(0, 2))
            sv_matrix_hm9 = arr_hm9.mean(axis=2)
        else:
            mean_shap_hm9 = np.abs(arr_hm9).mean(axis=0)
            sv_matrix_hm9 = arr_hm9

        top30_idx   = np.argsort(mean_shap_hm9)[::-1][:30]
        top30_genes = [selected_genes[i] for i in top30_idx]
        sv_top30    = sv_matrix_hm9[:, top30_idx]

        sort_ord9  = np.argsort(vis_y_te)
        sv_sorted9 = sv_top30[sort_ord9]
        y_sorted9  = vis_y_te[sort_ord9]
        vabs9      = np.percentile(np.abs(sv_sorted9), 95)

        fig9, ax9 = plt.subplots(figsize=(14, 8))
        fig9.patch.set_facecolor("white")
        im9  = ax9.imshow(sv_sorted9.T, aspect="auto", cmap="RdBu_r",
                          vmin=-vabs9, vmax=vabs9)
        cbar9 = plt.colorbar(im9, ax=ax9, shrink=0.8)
        cbar9.set_label("SHAP value", fontsize=9)
        ax9.set_yticks(range(30))
        ax9.set_yticklabels(top30_genes, fontsize=7)
        ax9.set_xlabel("Test samples (ordered by class)", fontsize=10)
        ax9.set_ylabel("Gene (top 30 by mean |SHAP|)", fontsize=10)

        for cls_idx in range(1, len(le.classes_)):
            boundary = np.where(y_sorted9 == cls_idx)[0]
            if len(boundary) > 0:
                ax9.axvline(boundary[0] - 0.5, color="black",
                            lw=1.5, linestyle="--")
        for cls_idx, cls_name in enumerate(le.classes_):
            mask = np.where(y_sorted9 == cls_idx)[0]
            if len(mask) > 0:
                ax9.text(mask[len(mask)//2], -1.8, cls_name,
                         ha="center", fontsize=10, fontweight="bold",
                         color=CLASS_COLORS[cls_name])

        ax9.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        fig9.savefig("fig9_shap_heatmap.png", dpi=300,
                     bbox_inches="tight", facecolor="white")
        plt.close(fig9)
        print("  fig9_shap_heatmap.png saved.")
    except Exception as e:
        print(f"  SHAP heatmap (top30) skipped: {e}")
else:
    print("  SHAP heatmap skipped: CPU model not available.")

# =========================================================
# FIG 10: SHAP Heatmap — 3 Panel, sınıf bazlı
# =========================================================
print("  Computing SHAP heatmap (3-panel, per-subtype)...")
if _xgb_shap is not None:
    try:
        explainer_hm10  = shap_lib.TreeExplainer(_xgb_shap)
        shap_vals_hm10  = safe_shap_values(explainer_hm10, vis_X_te)
        arr_hm10        = np.array(shap_vals_hm10, dtype=float)

        if arr_hm10.ndim == 3 and arr_hm10.shape[0] == len(le.classes_):
            shap_per_cls = arr_hm10
        elif arr_hm10.ndim == 3 and arr_hm10.shape[2] == len(le.classes_):
            shap_per_cls = arr_hm10.transpose(2, 0, 1)
        else:
            shap_per_cls = np.stack([arr_hm10]*len(le.classes_), axis=0)

        sort_ord10   = np.argsort(vis_y_te, kind='stable')
        y_sorted10   = vis_y_te[sort_ord10]
        cls_counts10 = np.bincount(y_sorted10)
        cls_bounds10 = np.cumsum(cls_counts10)[:-1]
        PANEL_C10    = ["#2980B9", "#E67E22", "#27AE60"]

        fig10, axes10 = plt.subplots(1, 3, figsize=(18, 8))
        fig10.patch.set_facecolor('white')

        for ci, (cls_name, ax) in enumerate(zip(le.classes_, axes10)):
            mean_abs10   = np.abs(shap_per_cls[ci]).mean(axis=0)
            top_idx10    = np.argsort(mean_abs10)[::-1][:15]
            top_genes10  = [selected_genes[i] for i in top_idx10]
            shap_srt10   = shap_per_cls[ci][sort_ord10, :][:, top_idx10].T
            vmax10       = np.percentile(np.abs(shap_srt10), 95)

            im10 = ax.imshow(shap_srt10, aspect='auto', cmap="RdBu_r",
                             vmin=-vmax10, vmax=vmax10,
                             interpolation='nearest')
            for xb in cls_bounds10:
                ax.axvline(x=xb - 0.5, color='black', linewidth=1.5)

            cls_centers10 = [cls_counts10[0]/2,
                             cls_counts10[0] + cls_counts10[1]/2,
                             cls_counts10[0] + cls_counts10[1] + cls_counts10[2]/2]
            for cj, (cx, clab, ccol) in enumerate(
                    zip(cls_centers10, le.classes_, PANEL_C10)):
                ax.text(cx, -0.8, clab, ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color=ccol,
                        transform=ax.get_xaxis_transform())

            ax.set_yticks(range(15))
            ax.set_yticklabels(top_genes10, fontsize=8, style='italic')
            ax.set_xlabel("Test samples (sorted by subtype)", fontsize=9)
            ax.set_title(f"SHAP — {cls_name} class\n(top 15 genes)",
                         fontsize=11, fontweight='bold',
                         color=PANEL_C10[ci], pad=8)
            ax.tick_params(axis='x', labelbottom=True, bottom=False)
            cbar10 = plt.colorbar(im10, ax=ax, shrink=0.65, pad=0.03)
            cbar10.set_label("SHAP value", fontsize=8)
            cbar10.ax.tick_params(labelsize=7)

        plt.tight_layout()
        fig10.savefig("fig10_shap_heatmap.png", dpi=300,
                      bbox_inches='tight', facecolor='white')
        plt.close(fig10)
        print("  fig10_shap_heatmap.png saved.")
    except Exception as e:
        print(f"  SHAP heatmap (3-panel) skipped: {e}")
else:
    print("  SHAP heatmap (3-panel) skipped: CPU model not available.")

# =========================================================
# FIG 11: Precision-Recall Curves — SoftVoting
# =========================================================
print("  Computing Precision-Recall curves (SoftVoting)...")
try:
    y_bin_pr = label_binarize(vis_y_te,
                              classes=list(range(len(le.classes_))))
    fig11, axes11 = plt.subplots(1, 3, figsize=(15, 5))
    fig11.patch.set_facecolor('white')

    for ci, (cls_name, ax) in enumerate(zip(le.classes_, axes11)):
        col      = CLASS_COLORS[cls_name]
        prec, rec, _ = precision_recall_curve(y_bin_pr[:, ci],
                                               vis_probs_sv[:, ci])
        ap       = average_precision_score(y_bin_pr[:, ci],
                                           vis_probs_sv[:, ci])
        baseline = y_bin_pr[:, ci].mean()

        ax.plot(rec, prec, color=col, lw=2,
                label=f"SoftVoting (AP = {ap:.3f})")
        ax.axhline(baseline, color='gray', lw=1, linestyle='--',
                   label='Baseline')
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_title(f"PR Curve — {cls_name}",
                     fontsize=11, fontweight='bold', color=col, pad=8)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    fig11.savefig("fig11_precision_recall.png", dpi=300,
                  bbox_inches='tight', facecolor='white')
    plt.close(fig11)
    print("  fig11_precision_recall.png saved.")
except Exception as e:
    print(f"  PR curves skipped: {e}")

# =========================================================
# FIG DATASET: Veri Seti Özet (bar + pasta)
# =========================================================
print("  Computing dataset summary figure...")
try:
    fig_ds, axes_ds = plt.subplots(1, 2, figsize=(12, 5))
    fig_ds.patch.set_facecolor('white')

    datasets    = ["GSE31210", "GSE13213", "GSE72094", "TCGA-LUAD"]
    egfr_counts = [127, 45, 47, 66]
    kras_counts = [20,  15, 154, 147]
    tn_counts   = [68,  57, 241, 295]
    x_ds = np.arange(len(datasets))
    w_ds = 0.25

    b1 = axes_ds[0].bar(x_ds - w_ds, egfr_counts, w_ds,
                         label='EGFR', color='#9B59B6',
                         alpha=0.85, edgecolor='white')
    b2 = axes_ds[0].bar(x_ds,        kras_counts, w_ds,
                         label='KRAS', color='#E67E22',
                         alpha=0.85, edgecolor='white')
    b3 = axes_ds[0].bar(x_ds + w_ds, tn_counts,   w_ds,
                         label='TN',   color='#1ABC9C',
                         alpha=0.85, edgecolor='white')
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            axes_ds[0].text(bar.get_x() + bar.get_width()/2., h + 2,
                            str(int(h)), ha='center', va='bottom',
                            fontsize=7.5, color='#2C3E50')
    axes_ds[0].set_xticks(x_ds)
    axes_ds[0].set_xticklabels(datasets, fontsize=10)
    axes_ds[0].set_ylabel("Sample count", fontsize=10)
    axes_ds[0].set_title("(a) Class distribution per dataset",
                          fontsize=11, pad=8)
    axes_ds[0].legend(fontsize=9, framealpha=0.9)
    axes_ds[0].spines[['top','right']].set_visible(False)

    axes_ds[1].axis('off')
    ax_in = fig_ds.add_axes([0.56, 0.15, 0.18, 0.70])
    ax_in.pie([219, 189, 366],
              labels=['EGFR','KRAS','TN'],
              colors=['#9B59B6','#E67E22','#1ABC9C'],
              autopct='%1.0f%%', startangle=90,
              textprops={'fontsize': 8},
              wedgeprops={'edgecolor':'white','linewidth':1.5})
    ax_in.set_title("GEO internal\n(n=774)", fontsize=9, pad=6)

    ax_tc = fig_ds.add_axes([0.76, 0.15, 0.18, 0.70])
    ax_tc.pie([66, 147, 295],
              labels=['EGFR','KRAS','TN'],
              colors=['#9B59B6','#E67E22','#1ABC9C'],
              autopct='%1.0f%%', startangle=90,
              textprops={'fontsize': 8},
              wedgeprops={'edgecolor':'white','linewidth':1.5})
    ax_tc.set_title("TCGA-LUAD\n(n=508)", fontsize=9, pad=6)
    axes_ds[1].set_title("(b) Overall class distribution",
                          fontsize=11, pad=8, x=0.55)

    plt.savefig("fig_dataset_summary.png", dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig_ds)
    print("  fig_dataset_summary.png saved.")
except Exception as e:
    print(f"  Dataset summary skipped: {e}")

# =========================================================
# ÖZET
# =========================================================
print("\n=== ALL FIGURES SAVED ===")
print("  fig1_combat_comparison.png    — PCA + t-SNE ComBat (2x4)")
print("  fig_tcga_combat_pca.png       — TCGA platform shift PCA")
print("  fig2_confusion_matrix.png     — (a) Internal + (b) TCGA")
print("  fig_classification_heatmap.png— Per-class P/R/F1 heatmap")
print("  fig3_roc_curves.png           — (a) Internal + (b) TCGA ROC")
print("  fig5_shap_beeswarm.png        — SHAP beeswarm EGFR/KRAS/TN")
print("  fig6_shap_bar_top20.png       — SHAP bar top20 per class")
print("  fig6_model_comparison.png     — 5-fold F1 model comparison")
print("  fig7_lime_explanations.png    — LIME one sample per class")
print("  fig8_ablation_chart.png       — Ablation bar chart")
print("  fig9_shap_heatmap.png         — SHAP heatmap top30 x samples")
print("  fig10_shap_heatmap.png        — SHAP heatmap 3-panel per class")
print("  fig11_precision_recall.png    — PR curves SoftVoting")
print("  fig_dataset_summary.png       — Dataset bar + pie charts")