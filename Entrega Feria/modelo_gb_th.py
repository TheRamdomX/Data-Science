import polars as pl
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import joblib
import shap
import gc

# =========================================================
# CONFIG
# =========================================================

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_PARQUET = os.path.join(BASE_DIR, "dataset_modelo.parquet")
RUTA_SALIDA         = os.path.join(BASE_DIR, "EDA_Resultados")
os.makedirs(RUTA_SALIDA, exist_ok=True)

RANDOM_SEED = 42
N_MUESTRA   = 2_000_000
MAX_DEPTH   = 12
N_FOLDS     = 5
THRESHOLDS  = np.arange(0.50, 0.96, 0.1)
SAVE_THRESHOLDS = [0.5, 0.7, 0.9]

FEATURES = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos",
    "fe_Region", "fe_Sector", "fe_TipoAdquisicion",
]

# =========================================================
# INFO
# =========================================================

print("\n" + "="*60)
print("GRADIENT BOOSTING -- THRESHOLD ANALYSIS (depth=12)")
print("="*60)

info = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .select([
        pl.len().alias("total"),
        pl.col("Target").cast(pl.Float64).mean().alias("prop_pos"),
    ])
    .collect()
)

total_filas = info["total"][0]
prop_pos    = info["prop_pos"][0]
del info; gc.collect()

print(f"  Filas totales    : {total_filas:,}")
print(f"  Target = 1       : {prop_pos:.4f} ({prop_pos*100:.1f}%)")
print(f"  max_depth        : {MAX_DEPTH}")
print(f"  Evaluacion       : {N_FOLDS}-fold Stratified CV (OOF)")
print(f"  Thresholds       : {len(THRESHOLDS)} valores ({THRESHOLDS[0]:.2f} - {THRESHOLDS[-1]:.2f})")
print(f"  Features         : {len(FEATURES)} (con NumOferentes)")

# =========================================================
# HELPERS
# =========================================================

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    gc.collect()
    print(f"  Guardado: {ruta}")


def build_scaler():
    stats_sc = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .select([
            *[pl.col(f).cast(pl.Float64).mean().alias(f"mean_{f}") for f in FEATURES],
            *[pl.col(f).cast(pl.Float64).std().alias(f"std_{f}")   for f in FEATURES],
        ])
        .collect()
    )
    scaler = StandardScaler()
    scaler.mean_  = np.array([stats_sc[f"mean_{f}"][0] or 0.0           for f in FEATURES])
    scaler.scale_ = np.array([max(stats_sc[f"std_{f}"][0] or 1.0, 1e-8) for f in FEATURES])
    scaler.n_features_in_ = len(FEATURES)
    del stats_sc; gc.collect()
    return scaler


def sample_data(n_sample):
    data_info = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .select([
            pl.len().alias("total"),
            pl.col("Target").cast(pl.Float64).mean().alias("prop"),
        ])
        .collect()
    )
    data_total = data_info["total"][0]
    data_prop  = data_info["prop"][0]
    del data_info

    n = min(n_sample, data_total)
    n_pos = int(data_prop * n)
    n_neg = n - n_pos
    print(f"  Muestra: {n:,} (pos={n_pos:,}, neg={n_neg:,})")

    muestra_pos = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Target") == 1)
        .collect().sample(n=min(n_pos, data_total), seed=RANDOM_SEED)
    )
    muestra_neg = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Target") == 0)
        .collect().sample(n=min(n_neg, data_total), seed=RANDOM_SEED)
    )
    muestra = pl.concat([muestra_pos, muestra_neg]).sample(fraction=1.0, seed=RANDOM_SEED)
    del muestra_pos, muestra_neg; gc.collect()

    X = np.nan_to_num(muestra.select(FEATURES).to_numpy(allow_copy=True).astype(np.float32))
    y = muestra["Target"].to_numpy().astype(np.int8)
    del muestra; gc.collect()
    return X, y


def metrics_at_threshold(y_true, y_proba, t):
    y_pred = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / cm.sum()
    return {"threshold": t, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "cm": cm}

# =========================================================
# CV + OOF
# =========================================================

label = "GB-A"
print(f"\n{'='*60}")
print(f"  {label}: {len(FEATURES)} features, depth={MAX_DEPTH}")
print(f"{'='*60}")

scaler = build_scaler()
X_raw, y_raw = sample_data(N_MUESTRA)
X_scaled = scaler.transform(X_raw)
del X_raw; gc.collect()

rus = RandomUnderSampler(random_state=RANDOM_SEED)
X_res, y_res = rus.fit_resample(X_scaled, y_raw)
del X_scaled, y_raw; gc.collect()
print(f"  Undersampling: {len(y_res):,} filas (pos={int(y_res.sum()):,})")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
oof_proba = np.zeros(len(y_res))
fold_aucs = []

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res), 1):
    X_tr, y_tr = X_res[train_idx], y_res[train_idx]
    X_val = X_res[val_idx]

    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=MAX_DEPTH,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )
    model.fit(X_tr, y_tr)
    oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_res[val_idx], oof_proba[val_idx])
    fold_aucs.append(fold_auc)
    print(f"    Fold {fold_i}: AUC={fold_auc:.4f} (iters={model.n_iter_})")
    del model, X_tr, y_tr, X_val; gc.collect()

oof_auc = roc_auc_score(y_res, oof_proba)
print(f"  OOF AUC: {oof_auc:.4f} (mean fold: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f})")

# =========================================================
# GRAFICO CROSS-VALIDATION
# =========================================================

fig, ax = plt.subplots(figsize=(8, 5))
folds_x = list(range(1, N_FOLDS + 1))
bars = ax.bar(folds_x, fold_aucs, color="#4472C4", edgecolor="white", width=0.6)
ax.axhline(np.mean(fold_aucs), color="red", linestyle="--", linewidth=1.5,
           label=f"Media = {np.mean(fold_aucs):.4f}")
ax.fill_between([0.5, N_FOLDS + 0.5],
                np.mean(fold_aucs) - np.std(fold_aucs),
                np.mean(fold_aucs) + np.std(fold_aucs),
                color="red", alpha=0.1, label=f"+/- std = {np.std(fold_aucs):.4f}")

for i, auc_val in enumerate(fold_aucs):
    ax.text(i + 1, auc_val + 0.001, f"{auc_val:.4f}", ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Fold", fontsize=11)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_title(f"{label} -- Cross-Validation ({N_FOLDS}-fold, depth={MAX_DEPTH})", fontsize=13)
ax.set_xticks(folds_x)
ax.set_xticklabels([f"Fold {i}" for i in folds_x])
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
y_min = min(fold_aucs) - 0.005
y_max = max(fold_aucs) + 0.005
ax.set_ylim(y_min, y_max)
fig.tight_layout()
guardar(fig, "modelo_GB_th_cv.png")

# =========================================================
# THRESHOLD ANALYSIS
# =========================================================

thresh_metrics = []
for t in THRESHOLDS:
    m = metrics_at_threshold(y_res, oof_proba, t)
    thresh_metrics.append(m)

best_f1_idx = int(np.argmax([m["f1"] for m in thresh_metrics]))
best_f1_t   = thresh_metrics[best_f1_idx]["threshold"]

print(f"\n  [{label}] Threshold optimo (max F1): {best_f1_t:.2f}")
print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
print(f"  {'-'*52}")
for m in thresh_metrics:
    marker = " <--" if abs(m["threshold"] - best_f1_t) < 0.001 else ""
    print(f"  {m['threshold']:>10.2f} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {m['accuracy']:>10.4f}{marker}")

y_pred_default = (oof_proba >= 0.5).astype(int)
print(f"\n  [{label}] Classification Report (threshold=0.50):")
print(classification_report(y_res, y_pred_default, digits=4))

y_pred_best = (oof_proba >= best_f1_t).astype(int)
print(f"  [{label}] Classification Report (threshold={best_f1_t:.2f}):")
print(classification_report(y_res, y_pred_best, digits=4))

# =========================================================
# MODELO FINAL: entrenar en todos los datos y guardar
# =========================================================

print(f"\n{'='*60}")
print(f"  Entrenando modelo final en todos los datos undersampled...")
print(f"{'='*60}")

final_model = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=MAX_DEPTH,
    learning_rate=0.05,
    min_samples_leaf=50,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
)
final_model.fit(X_res, y_res)
print(f"  Modelo final entrenado (iters={final_model.n_iter_})")

del X_res; gc.collect()

for t in SAVE_THRESHOLDS:
    t_str = f"{t:.1f}".replace(".", "")
    filename = f"modelo_gb_t{t_str}.joblib"
    filepath = os.path.join(RUTA_SALIDA, filename)

    m = metrics_at_threshold(y_res, oof_proba, t)

    bundle = {
        "model": final_model,
        "scaler": scaler,
        "threshold": t,
        "features": FEATURES,
        "max_depth": MAX_DEPTH,
        "oof_auc": oof_auc,
        "metrics_at_threshold": {
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
        },
    }
    joblib.dump(bundle, filepath)
    print(f"  Guardado: {filepath}  (threshold={t:.1f}, F1={m['f1']:.4f})")

# =========================================================
# SHAP VALUES
# =========================================================

print(f"\n{'='*60}")
print(f"  Calculando SHAP values...")
print(f"{'='*60}")

N_SHAP = 50_000
shap_sample = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .collect()
    .sample(n=N_SHAP, seed=RANDOM_SEED)
)
X_shap_raw = np.nan_to_num(shap_sample.select(FEATURES).to_numpy(allow_copy=True).astype(np.float32))
X_shap = scaler.transform(X_shap_raw)
del shap_sample, X_shap_raw; gc.collect()

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_shap)

shap.summary_plot(shap_values, X_shap, feature_names=FEATURES, show=False)
plt.title(f"SHAP Values -- Gradient Boosting (depth={MAX_DEPTH})", fontsize=13, fontweight="bold")
plt.tight_layout()
guardar(plt.gcf(), "modelo_GB_th_shap.png")

del explainer, shap_values, X_shap; gc.collect()
del final_model; gc.collect()

# =========================================================
# GRAFICO: Metricas vs Threshold
# =========================================================

ts   = [m["threshold"] for m in thresh_metrics]
prec = [m["precision"] for m in thresh_metrics]
rec  = [m["recall"] for m in thresh_metrics]
f1s  = [m["f1"] for m in thresh_metrics]
accs = [m["accuracy"] for m in thresh_metrics]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts, prec, "o-", color="#ED7D31", linewidth=2, markersize=8, label="Precision")
ax.plot(ts, rec,  "s-", color="#70AD47", linewidth=2, markersize=8, label="Recall")
ax.plot(ts, f1s,  "D-", color="#4472C4", linewidth=2, markersize=8, label="F1-Score")
ax.plot(ts, accs, "^-", color="#7F7F7F", linewidth=1.5, markersize=7, label="Accuracy", alpha=0.7)
ax.axvline(best_f1_t, color="red", linestyle="--", alpha=0.7,
           label=f"Mejor F1 (t={best_f1_t:.2f})")

for i, t in enumerate(ts):
    ax.annotate(f"{f1s[i]:.3f}", (t, f1s[i]), textcoords="offset points",
                xytext=(0, 10), fontsize=8, ha="center", color="#4472C4")

ax.set_xlabel("Threshold", fontsize=11)
ax.set_ylabel("Valor", fontsize=11)
ax.set_title(f"{label} -- Metricas vs Threshold (depth={MAX_DEPTH}, OOF AUC={oof_auc:.4f})",
             fontsize=12)
ax.legend(loc="best", fontsize=9)
ax.set_xlim(THRESHOLDS[0] - 0.02, THRESHOLDS[-1] + 0.02)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
guardar(fig, "modelo_GB_th.png")

# =========================================================
# CONFUSION MATRICES POR THRESHOLD
# =========================================================

show_ts = sorted(THRESHOLDS)
ncols = min(5, len(show_ts))
nrows = (len(show_ts) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()

for i, t in enumerate(show_ts):
    ax = axes[i]
    m = metrics_at_threshold(y_res, oof_proba, t)
    cm = m["cm"]
    total = cm.sum()

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for row in range(2):
        for col in range(2):
            count = cm[row, col]
            pct = 100.0 * count / total
            ax.text(col, row, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if count > cm.max() * 0.5 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Real 0", "Real 1"])
    tag = " *MEJOR*" if abs(t - best_f1_t) < 0.001 else ""
    ax.set_title(f"t={t:.2f}{tag}\nP={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}",
                 fontsize=9, fontweight="bold" if tag else "normal")

for j in range(len(show_ts), len(axes)):
    axes[j].axis("off")

fig.suptitle(f"{label} -- Matrices de Confusion por Threshold (depth={MAX_DEPTH})",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
guardar(fig, "modelo_GB_th_confusion.png")

# =========================================================
# CURVA ROC (OOF)
# =========================================================

fig, ax = plt.subplots(figsize=(8, 7))
fpr, tpr, _ = roc_curve(y_res, oof_proba)
ax.plot(fpr, tpr, linewidth=2, label=f"{label} (OOF AUC={oof_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title(f"Gradient Boosting -- Curva ROC (depth={MAX_DEPTH}, OOF)", fontsize=13)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.2)
fig.tight_layout()
guardar(fig, "modelo_GB_th_roc.png")

# =========================================================
# TABLA RESUMEN
# =========================================================

print("\n" + "="*60)
print("RESUMEN GRADIENT BOOSTING -- THRESHOLD ANALYSIS")
print("="*60)

m05 = metrics_at_threshold(y_res, oof_proba, 0.5)
mop = thresh_metrics[best_f1_idx]

print(f"  Modelo: {label}  |  OOF AUC: {oof_auc:.4f}  |  depth: {MAX_DEPTH}")
print(f"  Threshold default (0.50): P={m05['precision']:.4f}  R={m05['recall']:.4f}  F1={m05['f1']:.4f}")
print(f"  Threshold optimo ({best_f1_t:.2f}):  P={mop['precision']:.4f}  R={mop['recall']:.4f}  F1={mop['f1']:.4f}")

# Tabla imagen
fig, ax = plt.subplots(figsize=(12, 2.5))
ax.axis("off")
ax.set_title(f"Gradient Boosting -- Threshold Analysis (depth={MAX_DEPTH}, {N_FOLDS}-fold CV)",
             fontsize=13, fontweight="bold", pad=20)

cell_text = [[
    f"{oof_auc:.4f}",
    f"{m05['precision']:.4f}", f"{m05['recall']:.4f}", f"{m05['f1']:.4f}",
    f"{best_f1_t:.2f}",
    f"{mop['precision']:.4f}", f"{mop['recall']:.4f}", f"{mop['f1']:.4f}",
]]
col_labels = ["OOF AUC", "P@0.5", "R@0.5", "F1@0.5",
              "T_opt", "P@opt", "R@opt", "F1@opt"]

table = ax.table(
    cellText=cell_text, rowLabels=[label], colLabels=col_labels,
    cellLoc="center", rowLoc="center", loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == -1:
        cell.set_facecolor("#D9E2F3")
        cell.set_text_props(fontweight="bold")

fig.tight_layout()
guardar(fig, "modelo_GB_th_tabla_resumen.png")

# =========================================================
# GUARDAR RESULTADOS .npz
# =========================================================

save_data = {}
key = "GB_A"
save_data[f"{key}_fpr"] = fpr
save_data[f"{key}_tpr"] = tpr
save_data[f"{key}_auc"] = np.array([oof_auc])
save_data[f"{key}_best_threshold"] = np.array([best_f1_t])

y_pred_05 = (oof_proba >= 0.5).astype(int)
save_data[f"{key}_cm"] = confusion_matrix(y_res, y_pred_05)
save_data[f"{key}_precision"] = np.array([precision_score(y_res, y_pred_05)])
save_data[f"{key}_recall"] = np.array([recall_score(y_res, y_pred_05)])
save_data[f"{key}_f1"] = np.array([f1_score(y_res, y_pred_05)])

y_pred_opt = (oof_proba >= best_f1_t).astype(int)
save_data[f"{key}_cm_opt"] = confusion_matrix(y_res, y_pred_opt)
save_data[f"{key}_precision_opt"] = np.array([precision_score(y_res, y_pred_opt)])
save_data[f"{key}_recall_opt"] = np.array([recall_score(y_res, y_pred_opt)])
save_data[f"{key}_f1_opt"] = np.array([f1_score(y_res, y_pred_opt)])

save_data["labels"] = np.array([label])
np.savez(os.path.join(RUTA_SALIDA, "resultados_gb_th.npz"), **save_data)
print("\n  Resultados guardados en resultados_gb_th.npz")

print(f"\nGraficos en: {RUTA_SALIDA}")
print("\n" + "="*60)
print("COMPLETADO")
print("="*60)
