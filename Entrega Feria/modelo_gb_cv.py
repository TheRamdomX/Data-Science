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
import gc

# =========================================================
# CONFIG
# =========================================================

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_PARQUET = os.path.join(BASE_DIR, "dataset_modelo.parquet")
RUTA_SALIDA         = os.path.join(BASE_DIR, "EDA_Resultados")
os.makedirs(RUTA_SALIDA, exist_ok=True)

RANDOM_SEED = 42
N_MUESTRA   = 5_000_000

FEATURES_A = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos",
    "fe_Region", "fe_Sector", "fe_TipoAdquisicion",
]

FEATURES_B = [
    "MontoEstimado", "Valor Total Ofertado",
    "CantidadReclamos",
    "fe_Region", "fe_Sector", "fe_TipoAdquisicion",
]

MAX_DEPTHS  = [3, 5, 8, 12, 16, 20]
THRESHOLDS  = np.arange(0.10, 0.91, 0.05)
N_FOLDS     = 5

# =========================================================
# INFO
# =========================================================

print("\n" + "="*60)
print("GRADIENT BOOSTING -- CROSS-VALIDATION + THRESHOLD")
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
print(f"  Evaluacion       : {N_FOLDS}-fold Stratified CV (sin split temporal)")
print(f"  Profundidades    : {MAX_DEPTHS}")
print(f"  Thresholds       : {len(THRESHOLDS)} valores ({THRESHOLDS[0]:.2f} - {THRESHOLDS[-1]:.2f})")

# =========================================================
# HELPERS
# =========================================================

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    gc.collect()
    print(f"  Guardado: {ruta}")


def build_scaler(features):
    stats_sc = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .select([
            *[pl.col(f).cast(pl.Float64).mean().alias(f"mean_{f}") for f in features],
            *[pl.col(f).cast(pl.Float64).std().alias(f"std_{f}")   for f in features],
        ])
        .collect()
    )
    scaler = StandardScaler()
    scaler.mean_  = np.array([stats_sc[f"mean_{f}"][0] or 0.0           for f in features])
    scaler.scale_ = np.array([max(stats_sc[f"std_{f}"][0] or 1.0, 1e-8) for f in features])
    scaler.n_features_in_ = len(features)
    del stats_sc; gc.collect()
    return scaler


def sample_data(features, n_sample):
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

    X = np.nan_to_num(muestra.select(features).to_numpy(allow_copy=True).astype(np.float32))
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
# CROSS-VALIDATION: BUSQUEDA DE max_depth
# =========================================================

cv_results_all = {}

for feat_label, features in [("A", FEATURES_A), ("B", FEATURES_B)]:
    feat_desc = "Con NumOferentes" if feat_label == "A" else "Sin NumOferentes"
    print(f"\n{'='*60}")
    print(f"  CV -- Feature set {feat_label}: {feat_desc} ({len(features)} features)")
    print(f"{'='*60}")

    scaler = build_scaler(features)
    X_raw, y_raw = sample_data(features, N_MUESTRA)
    X_scaled = scaler.transform(X_raw)
    del X_raw; gc.collect()

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X_scaled, y_raw)
    del X_scaled, y_raw; gc.collect()
    print(f"  Undersampling: {len(y_res):,} filas (pos={int(y_res.sum()):,})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    depth_scores = {}

    for depth in MAX_DEPTHS:
        print(f"\n  max_depth={depth:>2} | {N_FOLDS}-fold CV ...", flush=True)
        fold_aucs = []

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res), 1):
            X_tr, y_tr = X_res[train_idx], y_res[train_idx]
            X_val, y_val = X_res[val_idx], y_res[val_idx]

            model = HistGradientBoostingClassifier(
                max_iter=500,
                max_depth=depth,
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
            y_proba_val = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba_val)
            fold_aucs.append(auc)
            print(f"    Fold {fold_i}: AUC={auc:.4f} (iters={model.n_iter_})")
            del model, X_tr, y_tr, X_val, y_val, y_proba_val; gc.collect()

        mean_auc = np.mean(fold_aucs)
        std_auc  = np.std(fold_aucs)
        depth_scores[depth] = {"mean": mean_auc, "std": std_auc, "folds": fold_aucs}
        print(f"    >> mean={mean_auc:.4f} +/- {std_auc:.4f}")

    best_depth = max(depth_scores, key=lambda d: depth_scores[d]["mean"])
    best_mean  = depth_scores[best_depth]["mean"]
    print(f"\n  Mejor max_depth={best_depth} (CV AUC={best_mean:.4f})")

    # --- OOF predictions con mejor depth para threshold analysis ---
    print(f"\n  Generando predicciones OOF (depth={best_depth})...")
    oof_proba = np.zeros(len(y_res))
    oof_iters = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_res, y_res), 1):
        X_tr, y_tr = X_res[train_idx], y_res[train_idx]
        X_val = X_res[val_idx]

        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=best_depth,
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
        oof_iters.append(model.n_iter_)
        del model, X_tr, y_tr, X_val; gc.collect()

    oof_auc = roc_auc_score(y_res, oof_proba)
    print(f"  OOF AUC: {oof_auc:.4f} (iters por fold: {oof_iters})")

    cv_results_all[feat_label] = {
        "feat_desc": feat_desc, "features": features, "scaler": scaler,
        "depth_scores": depth_scores, "best_depth": best_depth,
        "X_res": X_res, "y_res": y_res,
        "oof_proba": oof_proba, "oof_auc": oof_auc,
    }

# =========================================================
# GRAFICO CV: AUC vs max_depth
# =========================================================

fig, ax = plt.subplots(figsize=(10, 6))
for feat_label, data in cv_results_all.items():
    ds = data["depth_scores"]
    depths = sorted(ds.keys())
    means = [ds[d]["mean"] for d in depths]
    stds  = [ds[d]["std"] for d in depths]

    ax.errorbar(depths, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                label=f"Set {feat_label}: {data['feat_desc']} (mejor={data['best_depth']})")

    best_idx = depths.index(data["best_depth"])
    ax.annotate(f"depth={data['best_depth']}\nAUC={means[best_idx]:.4f}",
                xy=(depths[best_idx], means[best_idx]),
                xytext=(15, 15), textcoords="offset points",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))

ax.set_xlabel("max_depth", fontsize=11)
ax.set_ylabel("ROC-AUC (CV)", fontsize=11)
ax.set_title(f"Gradient Boosting -- AUC vs max_depth ({N_FOLDS}-fold CV, undersampling)", fontsize=13)
ax.set_xticks(MAX_DEPTHS)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
guardar(fig, "modelo_GB_cv_depth.png")

# =========================================================
# HEATMAP CV: folds x depths
# =========================================================

for feat_label, data in cv_results_all.items():
    ds = data["depth_scores"]
    depths = sorted(ds.keys())
    fold_matrix = np.array([ds[d]["folds"] for d in depths])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(fold_matrix, cmap="YlGn", aspect="auto")
    for i in range(len(depths)):
        for j in range(N_FOLDS):
            ax.text(j, i, f"{fold_matrix[i, j]:.4f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(N_FOLDS))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"depth={d}" for d in depths])
    ax.set_title(f"GB Set {feat_label} -- AUC por Fold y Profundidad", fontsize=12)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    guardar(fig, f"modelo_GB_cv_heatmap_{feat_label}.png")

# =========================================================
# ANALISIS DE THRESHOLD (sobre predicciones OOF)
# =========================================================

final_results = []

for feat_label, data in cv_results_all.items():
    best_depth = data["best_depth"]
    label      = f"GB-{feat_label}"
    y_true     = data["y_res"]
    y_proba    = data["oof_proba"]
    auc        = data["oof_auc"]

    print(f"\n{'='*60}")
    print(f"  ANALISIS: {label} (depth={best_depth}, OOF AUC={auc:.4f})")
    print(f"{'='*60}")

    thresh_metrics = []
    for t in THRESHOLDS:
        m = metrics_at_threshold(y_true, y_proba, t)
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

    y_pred_default = (y_proba >= 0.5).astype(int)
    print(f"\n  [{label}] Classification Report (threshold=0.50):")
    print(classification_report(y_true, y_pred_default, digits=4))

    y_pred_best = (y_proba >= best_f1_t).astype(int)
    print(f"  [{label}] Classification Report (threshold={best_f1_t:.2f}):")
    print(classification_report(y_true, y_pred_best, digits=4))

    final_results.append({
        "label": label, "feat_label": feat_label, "auc": auc,
        "best_depth": best_depth, "y_true": y_true, "y_proba": y_proba,
        "thresh_metrics": thresh_metrics, "best_f1_threshold": best_f1_t,
        "best_f1_metrics": thresh_metrics[best_f1_idx],
    })

# Liberar datos CV
for data in cv_results_all.values():
    del data["X_res"], data["y_res"], data["oof_proba"]
del cv_results_all; gc.collect()

# =========================================================
# GRAFICOS THRESHOLD
# =========================================================

for r in final_results:
    label = r["label"]
    tm    = r["thresh_metrics"]
    best_t = r["best_f1_threshold"]

    ts   = [m["threshold"] for m in tm]
    prec = [m["precision"] for m in tm]
    rec  = [m["recall"] for m in tm]
    f1s  = [m["f1"] for m in tm]
    accs = [m["accuracy"] for m in tm]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, prec, "o-", color="#ED7D31", linewidth=2, label="Precision")
    ax.plot(ts, rec,  "s-", color="#70AD47", linewidth=2, label="Recall")
    ax.plot(ts, f1s,  "D-", color="#4472C4", linewidth=2, label="F1-Score")
    ax.plot(ts, accs, "^-", color="#7F7F7F", linewidth=1.5, label="Accuracy", alpha=0.7)
    ax.axvline(best_t, color="red", linestyle="--", alpha=0.7,
               label=f"Mejor F1 (t={best_t:.2f})")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Default (t=0.50)")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title(f"{label} -- Metricas vs Threshold (depth={r['best_depth']}, OOF AUC={r['auc']:.4f})",
                 fontsize=12)
    ax.legend(loc="center left", fontsize=9)
    ax.set_xlim(THRESHOLDS[0] - 0.02, THRESHOLDS[-1] + 0.02)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    guardar(fig, f"modelo_GB_cv_threshold_{r['feat_label']}.png")

# =========================================================
# CONFUSION MATRICES: comparar thresholds seleccionados
# =========================================================

selected_thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

for r in final_results:
    label  = r["label"]
    best_t = r["best_f1_threshold"]
    show_ts = sorted(set(selected_thresholds + [round(best_t, 2)]))

    ncols = min(4, len(show_ts))
    nrows = (len(show_ts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, t in enumerate(show_ts):
        ax = axes[i]
        m = metrics_at_threshold(r["y_true"], r["y_proba"], t)
        cm = m["cm"]
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = 100.0 * count / total
                ax.text(col, row, f"{count:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if count > cm.max() * 0.5 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Real 0", "Real 1"])
        tag = " *MEJOR*" if abs(t - best_t) < 0.001 else ""
        ax.set_title(f"t={t:.2f}{tag}\nP={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}",
                     fontsize=9, fontweight="bold" if tag else "normal")

    for j in range(len(show_ts), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{label} -- Matrices de Confusion por Threshold (depth={r['best_depth']})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    guardar(fig, f"modelo_GB_cv_confusion_thresholds_{r['feat_label']}.png")

# =========================================================
# CURVAS ROC (OOF)
# =========================================================

fig, ax = plt.subplots(figsize=(8, 7))
for r in final_results:
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
    ax.plot(fpr, tpr, linewidth=2,
            label=f"{r['label']} depth={r['best_depth']} (OOF AUC={r['auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Gradient Boosting CV -- Curvas ROC (OOF)", fontsize=13)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.2)
fig.tight_layout()
guardar(fig, "modelo_GB_cv_roc.png")

# =========================================================
# TABLA RESUMEN
# =========================================================

print("\n" + "="*60)
print("RESUMEN GRADIENT BOOSTING -- CV + THRESHOLD")
print("="*60)

col_header = (f"  {'Modelo':<10} {'Depth':>6} {'AUC':>8} "
              f"{'T_def':>6} {'P@0.5':>8} {'R@0.5':>8} {'F1@0.5':>8} "
              f"{'T_opt':>6} {'P@opt':>8} {'R@opt':>8} {'F1@opt':>8}")
print(col_header)
print(f"  {'-'*96}")

for r in final_results:
    m05 = metrics_at_threshold(r["y_true"], r["y_proba"], 0.5)
    mop = r["best_f1_metrics"]
    print(f"  {r['label']:<10} {r['best_depth']:>6} {r['auc']:>8.4f} "
          f"{'0.50':>6} {m05['precision']:>8.4f} {m05['recall']:>8.4f} {m05['f1']:>8.4f} "
          f"{r['best_f1_threshold']:>6.2f} {mop['precision']:>8.4f} {mop['recall']:>8.4f} {mop['f1']:>8.4f}")

# =========================================================
# TABLA RESUMEN (imagen)
# =========================================================

labels_t = [r["label"] for r in final_results]
cell_text = []
for r in final_results:
    m05 = metrics_at_threshold(r["y_true"], r["y_proba"], 0.5)
    mop = r["best_f1_metrics"]
    cell_text.append([
        str(r["best_depth"]),
        f"{r['auc']:.4f}",
        f"{m05['precision']:.4f}", f"{m05['recall']:.4f}", f"{m05['f1']:.4f}",
        f"{r['best_f1_threshold']:.2f}",
        f"{mop['precision']:.4f}", f"{mop['recall']:.4f}", f"{mop['f1']:.4f}",
    ])

col_labels = ["Depth", "OOF AUC", "P@0.5", "R@0.5", "F1@0.5",
              "T_opt", "P@opt", "R@opt", "F1@opt"]

fig, ax = plt.subplots(figsize=(14, 1.5 + 0.6 * len(labels_t)))
ax.axis("off")
ax.set_title("Gradient Boosting -- Resumen CV + Threshold Optimo (sin split temporal)",
             fontsize=13, fontweight="bold", pad=20)

table = ax.table(
    cellText=cell_text, rowLabels=labels_t, colLabels=col_labels,
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
guardar(fig, "modelo_GB_cv_tabla_resumen.png")

# =========================================================
# GUARDAR RESULTADOS .npz
# =========================================================

save_data = {}
save_labels = []
for r in final_results:
    key = r["label"].replace("-", "_")
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
    save_data[f"{key}_fpr"] = fpr
    save_data[f"{key}_tpr"] = tpr
    save_data[f"{key}_auc"] = np.array([r["auc"]])
    save_data[f"{key}_best_depth"] = np.array([r["best_depth"]])
    save_data[f"{key}_best_threshold"] = np.array([r["best_f1_threshold"]])

    y_pred_05 = (r["y_proba"] >= 0.5).astype(int)
    cm05 = confusion_matrix(r["y_true"], y_pred_05)
    save_data[f"{key}_cm"] = cm05
    save_data[f"{key}_precision"] = np.array([precision_score(r["y_true"], y_pred_05)])
    save_data[f"{key}_recall"] = np.array([recall_score(r["y_true"], y_pred_05)])
    save_data[f"{key}_f1"] = np.array([f1_score(r["y_true"], y_pred_05)])

    y_pred_opt = (r["y_proba"] >= r["best_f1_threshold"]).astype(int)
    cm_opt = confusion_matrix(r["y_true"], y_pred_opt)
    save_data[f"{key}_cm_opt"] = cm_opt
    save_data[f"{key}_precision_opt"] = np.array([precision_score(r["y_true"], y_pred_opt)])
    save_data[f"{key}_recall_opt"] = np.array([recall_score(r["y_true"], y_pred_opt)])
    save_data[f"{key}_f1_opt"] = np.array([f1_score(r["y_true"], y_pred_opt)])

    save_labels.append(r["label"])

save_data["labels"] = np.array(save_labels)
np.savez(os.path.join(RUTA_SALIDA, "resultados_gb_cv.npz"), **save_data)
print("\n  Resultados guardados en resultados_gb_cv.npz")

print(f"\nGraficos en: {RUTA_SALIDA}")
print("\n" + "="*60)
print("COMPLETADO")
print("="*60)
