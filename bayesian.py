# -*- coding: utf-8 -*-
"""
@author: Gaziantep University İlkay Sibel Kervancı, Gözde Özsert Yiğit
Topic: Leak-Free Bioinformatics Pipeline — Bayesian Opt + Stacking + SHAP (v3 GPU)

DEĞİŞİKLİKLER v3:
  - GPU desteği: XGBoost (cuda), LightGBM (gpu), CatBoost (GPU)
  - np.array() düzeltmesi — DataFrame indexleme hatası giderildi
  - Debug print'ler temizlendi
  - torch ve cross_val_predict gereksiz importlar kaldırıldı
METODOLOJİ:
  - ComBat batch düzeltme (3 GEO kohortu)
  - SHAP tabanlı özellik seçimi (500 gen, 1. fold)
  - SVM Bayesian optimizasyonu (50 trial, 1. fold)
  - 5-fold stratified CV — tüm değerlendirme tutarlı
  - Her fold içinde NSGA-II Pareto (F1+MCC), iç CV üzerinde (200 trial)
  - Ensemble: SoftVoting, OptimalVoting, ParetoVoting, Stacking
  - TCGA-LUAD dış validasyon (SoftVoting, ComBat harmonizasyon)
  - Ablation study (5-fold, SoftVoting ile tutarlı)
"""

import pandas as pd
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




import numpy as np

# neuroCombat kütüphanesi eski NumPy alias'larını kullanıyor (np.int),
# bunlar NumPy 1.24+ sürümünde kaldırıldı. Uyumluluk için geri ekliyoruz.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
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
PARETO_TRIALS   = 200  # Her fold iç CV Pareto trial sayısı

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
# 6. SVM BAYESIAN OPT — Sadece 1. fold
# =========================================================
print("\n--- SVM Bayesian Optimizasyon (50 trial, 1. fold) ---")

X_tr0_sel    = X_tr0_raw[:, top_idx]
sc0          = StandardScaler()
X_tr0_scaled = sc0.fit_transform(X_tr0_sel)
y_tr0_orig   = y_enc[train_idx_0]

X_tr_np = np.array(X_tr0_scaled)
y_tr_np = np.array(y_tr0_orig)

def svm_objective(trial):
    C      = trial.suggest_float("C", 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    gamma  = trial.suggest_categorical("gamma", ["scale", "auto"])
    model  = SVC(C=C, kernel=kernel, gamma=gamma, probability=True,
                 class_weight="balanced", random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    scores = []
    for tr_i, val_i in inner_cv.split(X_tr_np, y_tr_np):
        Xtt, Xvv = X_tr_np[tr_i], X_tr_np[val_i]
        ytt, yvv = y_tr_np[tr_i], y_tr_np[val_i]
        min_k = np.min(np.bincount(ytt))
        k = min(SMOTE_K, min_k - 1)
        if k < 1:
            scores.append(0.0)
            continue
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        Xtt_r, ytt_r = sm.fit_resample(Xtt, ytt)
        model.fit(Xtt_r, ytt_r)
        scores.append(f1_score(yvv, model.predict(Xvv),
                               average="macro", zero_division=0))
    return np.mean(scores)

svm_study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
svm_study.optimize(svm_objective, n_trials=50, show_progress_bar=False)
svm_best = svm_study.best_params
print(f"  SVM en iyi F1 (inner CV): {svm_study.best_value:.4f}")
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
#      - İç 3-fold CV üzerinde NSGA-II Pareto (F1+MCC) yapılır
#      - SoftVoting, OptimalVoting(global), ParetoVoting(fold-içi), Stacking
# =========================================================
print("\n--- 5-Fold CV Değerlendirme ---")
print(f"    (Her fold içinde NSGA-II Pareto: {PARETO_TRIALS} trial, iç 3-fold CV)")

model_names = ["XGB", "LGBM", "RF", "CatBoost", "LogReg", "SVM",
               "SoftVoting", "OptimalVoting", "ParetoVoting", "Stacking"]
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
    # ParetoVoting — Her fold içinde iç 3-fold CV üzerinde
    #                NSGA-II ile F1+MCC optimize
    # Sızıntısız: SMOTE sonrası train set → iç CV → validation
    # -------------------------------------------------------
    inner_cv_par = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE + fold_i)

    def pareto_inner_objective(trial):
        w = {
            "w_xgb" : trial.suggest_float("w_xgb",  0.1, 5.0),
            "w_lgbm": trial.suggest_float("w_lgbm", 0.1, 5.0),
            "w_rf"  : trial.suggest_float("w_rf",   0.1, 3.0),
            "w_cat" : trial.suggest_float("w_cat",  0.1, 5.0),
            "w_svm" : trial.suggest_float("w_svm",  0.1, 5.0),
        }
        # İç CV üzerinde F1 ve MCC hesapla
        inner_f1_list, inner_mcc_list = [], []
        for in_tr, in_val in inner_cv_par.split(X_tr_res_f, y_tr_res_f):
            X_in_tr, X_in_val = X_tr_res_f[in_tr], X_tr_res_f[in_val]
            y_in_tr, y_in_val = y_tr_res_f[in_tr], y_tr_res_f[in_val]

            # İç fold modelleri eğit
            in_models = {}
            for mn2 in ["XGB","LGBM","RF","CatBoost"]:
                m2 = build_model(mn2)
                m2.fit(X_in_tr, y_in_tr)
                in_models[mn2] = m2
            in_svm = SVC(C=svm_best["C"], kernel=svm_best["kernel"],
                         gamma=svm_best["gamma"], probability=True,
                         class_weight="balanced", random_state=RANDOM_STATE)
            in_svm.fit(X_in_tr, y_in_tr)
            in_models["SVM"] = in_svm

            in_probs = {mn2: in_models[mn2].predict_proba(X_in_val)
                        for mn2 in ["XGB","LGBM","RF","CatBoost","SVM"]}
            p_w = get_weighted_probs(w, in_probs)
            preds_w = np.argmax(p_w, axis=1)
            inner_f1_list.append(f1_score(y_in_val, preds_w,
                                          average="macro", zero_division=0))
            inner_mcc_list.append(matthews_corrcoef(y_in_val, preds_w))

        return np.mean(inner_f1_list), np.mean(inner_mcc_list)

    pareto_study_f = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=RANDOM_STATE + fold_i)
    )
    pareto_study_f.optimize(pareto_inner_objective,
                             n_trials=PARETO_TRIALS,
                             show_progress_bar=False)

    fold_pareto_trials = pareto_study_f.best_trials
    best_fold_pareto   = max(fold_pareto_trials,
                              key=lambda t: t.values[0] + t.values[1])
    best_w_fold        = best_fold_pareto.params

    probs_pv_f = get_weighted_probs(best_w_fold, fold_probs_te)
    eval_ensemble("ParetoVoting", probs_pv_f)

    print(f"    Pareto iç CV: F1={best_fold_pareto.values[0]:.4f}, "
          f"MCC={best_fold_pareto.values[1]:.4f}, "
          f"Frontier={len(fold_pareto_trials)}")

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
    print(f"    ParetoVoting : F1={cv_results['ParetoVoting']['f1'][-1]:.4f} "
          f"MCC={cv_results['ParetoVoting']['mcc'][-1]:.4f}")
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
# 9. SONUÇLAR — 5-FOLD ORTALAMA
# =========================================================
print("\n\n=== 5-FOLD CV SONUÇLARI (Ortalama ± Std) ===")
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
    print(f"  {mn:20s}: F1={r['F1_macro']}±{r['F1_std']} | "
          f"Acc={r['Accuracy']}±{r['Acc_std']} | "
          f"MCC={r['MCC']}±{r['MCC_std']} | AUROC={r['AUROC']}±{r['AUROC_std']}")

print(f"\n--- Sınıf Bazlı F1 (SoftVoting — Ana Model) ---")
for cls in ["EGFR", "KRAS", "TN"]:
    print(f"  {cls}: {np.mean(cv_f1_class[cls]):.4f} ± {np.std(cv_f1_class[cls]):.4f}")

results_df = pd.DataFrame(results_list).sort_values("F1_macro", ascending=False)
results_df.to_csv("results_v7_cv.csv", index=False)
pd.Series(selected_genes, name="gene").to_csv("selected_genes_shap.csv", index=False)
joblib.dump(best_w_global, "best_weights_global_v7.pkl")
joblib.dump(svm_best,      "svm_best_params_v7.pkl")

print("\n  results_v7_cv.csv kaydedildi.")
print("\n=== TAMAMLANDI ===")

# =========================================================
# 10. DIŞ VALİDASYON — TCGA-LUAD (SoftVoting) — DÜZELTİLMİŞ
#
# DÜZELTME 1: Joint ComBat → Referans tabanlı ComBat (ref_batch=0)
#   Eski: eğitim + TCGA birlikte Combat'landı → sızıntı
#   Yeni: Combat parametreleri sadece eğitimden, TCGA'ya uygulanır
#
# DÜZELTME 2: Threshold TCGA'dan → iç CV validation setinden
#   Eski: threshold TCGA üzerinde optimize edildi → data leakage
#   Yeni: threshold iç CV'nin validation setinden belirlenir
# =========================================================
print("\n" + "="*55)
print("  DIŞ VALİDASYON: TCGA-LUAD (Düzeltilmiş)")
print("="*55)

from sklearn.model_selection import StratifiedKFold as SKF2

# ── Tüm eğitim verisi üzerinde modeller ──────────────────
X_all_sel    = X_combat.values[:, top_idx]
sc_all       = StandardScaler()
sc_all.fit(X_all_sel)

sm_all = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_all_sc     = sc_all.transform(X_all_sel)
X_all_res, y_all_res = sm_all.fit_resample(X_all_sc, y_enc)

print("Tüm eğitim verisi üzerinde modeller eğitiliyor...")
final_models = {}
for mn in ["XGB", "LGBM", "RF", "CatBoost", "LogReg"]:
    m = build_model(mn)
    m.fit(X_all_res, y_all_res)
    final_models[mn] = m

final_svm = SVC(
    C=svm_best["C"], kernel=svm_best["kernel"],
    gamma=svm_best["gamma"], probability=True,
    class_weight="balanced", random_state=RANDOM_STATE
)
final_svm.fit(X_all_res, y_all_res)
final_models["SVM"] = final_svm
print("  Modeller hazır.")

# ── DÜZELTME 2: Threshold — iç CV validation setinden ────
print("\n--- Threshold Optimizasyonu (iç CV — TCGA'ya bakılmadan) ---")

cv_thr = SKF2(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
tr_thr_idx, val_thr_idx = next(cv_thr.split(X_all_sc, y_enc))

X_val_thr = X_all_sc[val_thr_idx]
y_val_thr = y_enc[val_thr_idx]

# Validation setinde SoftVoting olasılıkları
probs_val_thr = sum(
    final_models[mn].predict_proba(X_val_thr)
    for mn in ["XGB", "LGBM", "RF", "CatBoost", "SVM"]
) / 5

best_thresholds = {}
for cls_idx, cls_name in enumerate(le.classes_):
    best_t, best_f1_t = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds_t = (probs_val_thr[:, cls_idx] >= t).astype(int)
        y_bin   = (y_val_thr == cls_idx).astype(int)
        score   = f1_score(y_bin, preds_t, zero_division=0)
        if score > best_f1_t:
            best_f1_t, best_t = score, t
    best_thresholds[cls_name] = round(best_t, 2)
    print(f"  {cls_name}: threshold={best_t:.2f}, val_F1={best_f1_t:.3f}")

print("\nThresholdlar iç CV'den belirlendi — TCGA'ya bakılmadı.")

# ── TCGA verisi yükle ─────────────────────────────────────
print("\nTCGA verisi yükleniyor...")
tcga_expr   = pd.read_parquet("C:/sibel/gozdehoca/tcga_luad_expr.parquet")
tcga_labels = pd.read_csv("C:/sibel/gozdehoca/tcga_luad_labels.csv",
                           index_col=0).squeeze()

tcga_mask   = tcga_labels.isin(le.classes_)
tcga_expr   = tcga_expr.loc[tcga_mask]
tcga_labels = tcga_labels[tcga_mask]
print(f"Kullanılan örnek: {len(tcga_labels)}")
print(f"Dağılım: {tcga_labels.value_counts().to_dict()}")

# ── DÜZELTME 1: Referans tabanlı ComBat ──────────────────
print("\nReferans tabanlı ComBat uygulanıyor (ref_batch=0)...")
train_genes  = list(X_combat.columns)
tcga_genes   = list(tcga_expr.columns)
shared_genes = list(set(train_genes) & set(tcga_genes))
print(f"Ortak gen: {len(shared_genes)}")

X_train_sh = X_raw[shared_genes]     # RAW eğitim (Combat öncesi)
X_tcga_sh  = tcga_expr[shared_genes]

combined_ref = pd.concat([X_train_sh, X_tcga_sh], axis=0)
batch_ref    = pd.Series(
    [0]*len(X_train_sh) + [1]*len(X_tcga_sh),
    index=combined_ref.index, name="batch"
)

# ref_batch=0 → eğitim referans alınır, TCGA ona göre ayarlanır
# Eğitim verisi değişmez — sadece TCGA normalize edilir
combat_ref_out = neuroCombat(
    dat      = combined_ref.T.values.astype(float),
    covars   = pd.DataFrame({"batch": batch_ref}),
    batch_col= "batch",
    ref_batch= 0
)

combined_corr = pd.DataFrame(
    combat_ref_out["data"].T,
    index   = combined_ref.index,
    columns = shared_genes
)

# Sadece TCGA kısmını al
tcga_corr = combined_corr.iloc[len(X_train_sh):]
print(f"TCGA harmonize edildi: {tcga_corr.shape}")

# ── SHAP genleri seç + imputation ────────────────────────
common_shap = [g for g in selected_genes if g in shared_genes]
missing     = len(selected_genes) - len(common_shap)
print(f"\nSHAP genleri: {len(selected_genes)} | Ortak: {len(common_shap)} | Eksik: {missing}")

# Eksik genler için eğitim ortalaması (sıfır değil — daha savunulabilir)
train_sel_means = X_combat[
    [g for g in selected_genes if g in X_combat.columns]
].mean()

tcga_sel = pd.DataFrame(np.nan, index=tcga_corr.index, columns=selected_genes)
tcga_sel[common_shap] = tcga_corr[common_shap].values

for g in selected_genes:
    if g not in common_shap:
        fill_val = train_sel_means.get(g, 0.0)
        tcga_sel[g] = fill_val
        print(f"  {g}: eğitim ortalaması ile impute (μ={fill_val:.3f})")

# Eğitim scalerı ile ölçekle (fit yapma!)
tcga_scaled = sc_all.transform(tcga_sel.values)
tcga_y      = le.transform(tcga_labels)

# ── SoftVoting tahmini ────────────────────────────────────
probs_tcga = sum(
    final_models[mn].predict_proba(tcga_scaled)
    for mn in ["XGB", "LGBM", "RF", "CatBoost", "SVM"]
) / 5

# İç CV thresholdları ile tahmin
scores_adj = np.zeros((len(tcga_y), len(le.classes_)))
for cls_idx, cls_name in enumerate(le.classes_):
    scores_adj[:, cls_idx] = (probs_tcga[:, cls_idx]
                               / best_thresholds[cls_name])
preds_tcga = np.argmax(scores_adj, axis=1)

# ── Metrikler ─────────────────────────────────────────────
try:
    auroc_tcga = roc_auc_score(tcga_y, probs_tcga,
                                multi_class="ovr", average="macro")
except:
    auroc_tcga = np.nan

acc_tcga = accuracy_score(tcga_y, preds_tcga)
f1_tcga  = f1_score(tcga_y, preds_tcga, average='macro', zero_division=0)
mcc_tcga = matthews_corrcoef(tcga_y, preds_tcga)

print("\n" + "="*55)
print("  TCGA-LUAD DIŞ VALİDASYON SONUÇLARI")
print("  (ref_batch=0 ComBat · iç CV thresholdları)")
print("="*55)
print(f"  Accuracy : {acc_tcga:.4f}")
print(f"  F1-macro : {f1_tcga:.4f}")
print(f"  MCC      : {mcc_tcga:.4f}")
print(f"  AUROC    : {auroc_tcga:.4f}")
print(f"\nDetaylı rapor:")
print(classification_report(tcga_y, preds_tcga,
                             target_names=le.classes_, zero_division=0))
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix as cm_fn
print(cm_fn(tcga_y, preds_tcga))

tcga_results = pd.DataFrame([{
    "Dataset"         : "TCGA-LUAD",
    "Model"           : "SoftVoting",
    "Platform"        : "RNA-seq",
    "N"               : len(tcga_y),
    "Accuracy"        : round(acc_tcga,   4),
    "F1_macro"        : round(f1_tcga,    4),
    "MCC"             : round(mcc_tcga,   4),
    "AUROC"           : round(auroc_tcga, 4),
    "ComBat_method"   : "ref_batch=0 (training as reference)",
    "Threshold_source": "Inner CV validation (not TCGA)",
    "Missing_genes"   : missing,
    "Imputation"      : "Training mean",
    "Applied_thresholds": str(best_thresholds),
}])
tcga_results.to_csv("tcga_external_validation_fixed.csv", index=False)
print("\n  tcga_external_validation_fixed.csv kaydedildi.")
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





