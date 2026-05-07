# -*- coding: utf-8 -*-
"""
@author: Gaziantep University İlkay Sibel Kervancı, Gözde Özsert Yiğit
Topic: Leak-Free Bioinformatics Pipeline — Bayesian Opt + Stacking + SHAP (v3 GPU)

DEĞİŞİKLİKLER v3:
  - GPU desteği: XGBoost (cuda), LightGBM (gpu), CatBoost (GPU)
  - np.array() düzeltmesi — DataFrame indexleme hatası giderildi
  - Debug print'ler temizlendi
  - torch ve cross_val_predict gereksiz importlar kaldırıldı
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               VotingClassifier)
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# 0. AYARLAR
# =========================================================
N_TRIALS        = 50
N_FEATURES_SHAP = 500
OUTER_FOLDS     = 5
INNER_FOLDS     = 3
RANDOM_STATE    = 42
SMOTE_K         = 5

# GPU kontrolü
USE_GPU = torch.cuda.is_available()
print(f"GPU durumu: {'AKTIF — ' + torch.cuda.get_device_name(0) if USE_GPU else 'YOK, CPU kullanılacak'}")

# =========================================================
# 1. VERİ YÜKLEME
# =========================================================
print("\nVeriler yükleniyor...")
X_raw = pd.read_parquet("expr_top2000_genes.parquet")
meta  = pd.read_csv("metadata_combined.csv", index_col=0)

# ALK çıkar — y_enc hesabından ÖNCE
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
# 2. COMBAT
# =========================================================
print("\n--- ComBat ---")
covars_df  = meta.loc[X_raw.index, ["batch"]].copy()
combat_out = neuroCombat(dat=X_raw.T.values, covars=covars_df, batch_col="batch")
X_combat   = pd.DataFrame(combat_out["data"].T, index=X_raw.index, columns=X_raw.columns)
print(f"ComBat tamamlandı. NaN: {X_combat.isna().sum().sum()}")

# =========================================================
# 3. METRİK FONKSİYONU
# =========================================================
def compute_metrics(y_true, y_pred, y_prob, model_name, extra=None):
    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auroc = np.nan
    d = {
        "Model"          : model_name,
        "Accuracy"       : round(accuracy_score(y_true, y_pred), 4),
        "Precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall_macro"   : round(recall_score(y_true, y_pred,    average="macro", zero_division=0), 4),
        "F1_macro"       : round(f1_score(y_true, y_pred,        average="macro", zero_division=0), 4),
        "F1_weighted"    : round(f1_score(y_true, y_pred,        average="weighted", zero_division=0), 4),
        "MCC"            : round(matthews_corrcoef(y_true, y_pred), 4),
        "AUROC"          : round(auroc, 4),
    }
    if extra:
        d.update(extra)
    return d

# =========================================================
# 4. MODEL OLUŞTURUCU — GPU DESTEKLİ
# =========================================================
def build_model(name, params):
    p = params.copy()
    if name == "XGB":
        p.update({
            "use_label_encoder": False,
            "eval_metric"      : "mlogloss",
            "random_state"     : RANDOM_STATE,
            "tree_method"      : "hist",
            "device"           : "cuda" if USE_GPU else "cpu",
        })
        return XGBClassifier(**p)

    elif name == "LGBM":
        p.update({
            "class_weight" : "balanced",
            "random_state" : RANDOM_STATE,
            "verbose"      : -1,
            "device"       : "gpu" if USE_GPU else "cpu",
        })
        return LGBMClassifier(**p)

    elif name == "RF":
        # sklearn RF — GPU yok, n_jobs=-1 ile CPU paralel
        p.update({
            "class_weight" : "balanced",
            "random_state" : RANDOM_STATE,
            "n_jobs"       : -1,
        })
        return RandomForestClassifier(**p)

    elif name == "CatBoost":
        p.update({
            "auto_class_weights": "Balanced",
            "random_seed"       : RANDOM_STATE,
            "verbose"           : 0,
            "task_type"         : "GPU" if USE_GPU else "CPU",
        })
        return CatBoostClassifier(**p)

    elif name == "LogReg":
        p.update({
            "max_iter"     : 2000,
            "class_weight" : "balanced",
            "multi_class"  : "multinomial",
            "random_state" : RANDOM_STATE,
        })
        return LogisticRegression(**p)

# =========================================================
# 5. BAYESIAN OPT — OBJECTIVE FONKSİYONU
# =========================================================
def get_objective(model_name, X_tr, y_tr):
    """Her model için Optuna objective döndürür."""

    # X_tr kesinlikle numpy array olmalı
    X_tr = np.array(X_tr)
    y_tr = np.array(y_tr)

    def objective(trial):
        if model_name == "XGB":
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 600),
                max_depth        = trial.suggest_int("max_depth", 3, 10),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
                gamma            = trial.suggest_float("gamma", 0, 5),
                reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            )
        elif model_name == "LGBM":
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 600),
                max_depth        = trial.suggest_int("max_depth", 3, 12),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves       = trial.suggest_int("num_leaves", 20, 200),
                subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
                min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
                reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            )
        elif model_name == "RF":
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 100, 600),
                max_depth         = trial.suggest_int("max_depth", 3, 30),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 10),
                max_features      = trial.suggest_categorical("max_features",
                                        ["sqrt", "log2", 0.3, 0.5]),
            )
        elif model_name == "CatBoost":
            params = dict(
                iterations          = trial.suggest_int("iterations", 100, 600),
                depth               = trial.suggest_int("depth", 3, 10),
                learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1, 20.0),
                border_count        = trial.suggest_int("border_count", 32, 255),
                bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
            )
        elif model_name == "LogReg":
            params = dict(
                C      = trial.suggest_float("C", 1e-4, 100.0, log=True),
                solver = trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            )

        model = build_model(model_name, params)

        # İç CV
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                                    random_state=RANDOM_STATE)
        scores = []
        for train_i, val_i in inner_cv.split(X_tr, y_tr):
            Xtt, Xvv = X_tr[train_i], X_tr[val_i]
            ytt, yvv = y_tr[train_i], y_tr[val_i]
            min_class_count = np.min(np.bincount(ytt))
            k = min(SMOTE_K, min_class_count - 1)
            if k < 1:
                scores.append(0.0)
                continue
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            Xtt_r, ytt_r = sm.fit_resample(Xtt, ytt)
            model.fit(Xtt_r, ytt_r)
            preds = model.predict(Xvv)
            scores.append(f1_score(yvv, preds, average="macro", zero_division=0))
        return np.mean(scores)

    return objective

# =========================================================
# 6. SHAP FEATURE SELECTION
# =========================================================
print(f"\n--- SHAP Feature Selection ({N_FEATURES_SHAP} gen) ---")
outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(outer_cv.split(X_combat.values, y_enc))

X_tr_raw = X_combat.values[train_idx]
X_te_raw = X_combat.values[test_idx]
y_tr     = y_enc[train_idx]
y_te     = y_enc[test_idx]

scaler_pre       = StandardScaler()
X_tr_pre         = scaler_pre.fit_transform(X_tr_raw)

sm_pre           = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_tr_pre_res, y_tr_pre_res = sm_pre.fit_resample(X_tr_pre, y_tr)

rf_shap = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
rf_shap.fit(X_tr_pre_res, y_tr_pre_res)

explainer = shap.TreeExplainer(rf_shap)
shap_vals = explainer.shap_values(X_tr_pre[:300])

if isinstance(shap_vals, list):
    mean_shap = np.zeros(X_tr_pre.shape[1])
    for sv in shap_vals:
        mean_shap += np.abs(np.array(sv)).mean(axis=0)
    mean_shap /= len(shap_vals)
else:
    arr = np.array(shap_vals)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes) — bu veri seti için bu format
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

# Seçilmiş feature'larla scale
X_tr_sel    = X_tr_raw[:, top_idx]
X_te_sel    = X_te_raw[:, top_idx]
scaler      = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr_sel)
X_te_scaled = scaler.transform(X_te_sel)

sm_main = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
X_tr_res, y_tr_res = sm_main.fit_resample(X_tr_scaled, y_tr)
print(f"SMOTE sonrası: {X_tr_res.shape[0]} örnek | Dağılım: {np.bincount(y_tr_res)}")

# =========================================================
# 7. BAYESIAN OPT — HER MODEL İÇİN
# =========================================================
model_names = ["XGB", "LGBM", "RF", "CatBoost", "LogReg"]
best_params = {}
best_models = {}

print("\n--- Bayesian Optimizasyon ---")
for mname in model_names:
    print(f"  {mname} optimize ediliyor ({N_TRIALS} trial)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(
        get_objective(mname, X_tr_scaled, y_tr),   # np.array() dönüşümü objective içinde
        n_trials=N_TRIALS,
        show_progress_bar=False
    )
    best_params[mname] = study.best_params
    print(f"    En iyi F1_macro (inner CV): {study.best_value:.4f}")
    print(f"    Parametreler: {study.best_params}")

# =========================================================
# 8. FİNAL MODELLER
# =========================================================
print("\n--- Final Modeller ---")
results_list = []

for mname in model_names:
    model = build_model(mname, best_params[mname])
    model.fit(X_tr_res, y_tr_res)
    best_models[mname] = model
    preds = model.predict(X_te_scaled)
    probs = model.predict_proba(X_te_scaled)
    r = compute_metrics(y_te, preds, probs, f"{mname}_BayesOpt",
                        extra={"N_features": N_FEATURES_SHAP, "N_trials": N_TRIALS})
    results_list.append(r)
    print(f"  {mname}: Acc={r['Accuracy']} | F1={r['F1_macro']} | "
          f"MCC={r['MCC']} | AUROC={r['AUROC']}")

# =========================================================
# 9. STACKING ENSEMBLE
# =========================================================
print("\n--- Stacking Ensemble ---")

base_estimators = [
    ("xgb",  build_model("XGB",      best_params["XGB"])),
    ("lgbm", build_model("LGBM",     best_params["LGBM"])),
    ("rf",   build_model("RF",       best_params["RF"])),
    ("cat",  build_model("CatBoost", best_params["CatBoost"])),
]

meta_lr = LogisticRegression(
    C=1.0, max_iter=2000, class_weight="balanced",
    multi_class="multinomial", random_state=RANDOM_STATE
)

stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_lr,
    cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    passthrough=True,
    stack_method="predict_proba",
    n_jobs=1,   # GPU modelleriyle n_jobs=-1 çakışabilir
)

stack_clf.fit(X_tr_res, y_tr_res)
preds_st = stack_clf.predict(X_te_scaled)
probs_st = stack_clf.predict_proba(X_te_scaled)

r_stack = compute_metrics(y_te, preds_st, probs_st, "Stacking_BayesOpt",
                           extra={"N_features": N_FEATURES_SHAP, "N_trials": N_TRIALS})
results_list.append(r_stack)
print(f"  Stacking: Acc={r_stack['Accuracy']} | F1={r_stack['F1_macro']} | "
      f"MCC={r_stack['MCC']} | AUROC={r_stack['AUROC']}")

# =========================================================
# 10. SOFT VOTING
# =========================================================
print("\n--- Soft Voting ---")

vote_clf = VotingClassifier(
    estimators=base_estimators,
    voting="soft",
    n_jobs=1,   # GPU modelleriyle n_jobs=-1 çakışabilir
)
vote_clf.fit(X_tr_res, y_tr_res)
preds_v = vote_clf.predict(X_te_scaled)
probs_v = vote_clf.predict_proba(X_te_scaled)

r_vote = compute_metrics(y_te, preds_v, probs_v, "SoftVoting_BayesOpt",
                          extra={"N_features": N_FEATURES_SHAP, "N_trials": N_TRIALS})
results_list.append(r_vote)
print(f"  SoftVoting: Acc={r_vote['Accuracy']} | F1={r_vote['F1_macro']} | "
      f"MCC={r_vote['MCC']} | AUROC={r_vote['AUROC']}")

# =========================================================
# 11. SONUÇLAR & KAYIT
# =========================================================
results_df = pd.DataFrame(results_list).sort_values("F1_macro", ascending=False)
print("\n=== BAYESIAN OPT + STACKING SONUÇLARI ===")
print(results_df.to_string(index=False))
results_df.to_csv("bayesian_results.csv", index=False)

# En iyi modeli kaydet
best_name = results_df.iloc[0]["Model"]
print(f"\nEn iyi model: {best_name}")

if "Stacking" in best_name:
    save_model = stack_clf
elif "SoftVoting" in best_name:
    save_model = vote_clf
else:
    save_model = best_models.get(best_name.replace("_BayesOpt", ""))

if save_model:
    joblib.dump(save_model, "best_model.pkl")
    print("  best_model.pkl kaydedildi.")

pd.Series(selected_genes, name="gene").to_csv("selected_genes_shap.csv", index=False)
print(f"  selected_genes_shap.csv kaydedildi ({len(selected_genes)} gen).")

params_df = pd.DataFrame([{"model": k, **v} for k, v in best_params.items()])
params_df.to_csv("bayesian_best_params.csv", index=False)
print("  bayesian_best_params.csv kaydedildi.")

print("\n=== TAMAMLANDI ===")
print("  bayesian_results.csv     — tüm model karşılaştırmaları")
print("  best_model.pkl           — en iyi model")
print("  selected_genes_shap.csv  — SHAP ile seçilen genler")
print("  bayesian_best_params.csv — optimizasyon parametreleri")


# =========================================================
# AĞIRLIKLI VOTING — XGB'ye fazla ağırlık
# =========================================================
print("\n--- Ağırlıklı Voting (XGB:3, LGBM:2, RF:1, CatBoost:2) ---")

weighted_vote = VotingClassifier(
    estimators=base_estimators,
    voting="soft",
    weights=[3, 2, 1, 2],  # xgb, lgbm, rf, catboost
    n_jobs=1,
)
weighted_vote.fit(X_tr_res, y_tr_res)
preds_wv = weighted_vote.predict(X_te_scaled)
probs_wv = weighted_vote.predict_proba(X_te_scaled)

r_wv = compute_metrics(y_te, preds_wv, probs_wv, "WeightedVoting_BayesOpt",
                        extra={"N_features": N_FEATURES_SHAP, "N_trials": N_TRIALS})
results_list.append(r_wv)
print(f"  WeightedVoting: Acc={r_wv['Accuracy']} | F1={r_wv['F1_macro']} | "
      f"MCC={r_wv['MCC']} | AUROC={r_wv['AUROC']}")
