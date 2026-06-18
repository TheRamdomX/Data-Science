import polars as pl
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
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
CHUNK_SIZE  = 500_000
N_MUESTRA   = 500_000

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

KNN_NEIGHBORS = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]

# =========================================================
# INFO + SPLIT
# =========================================================

print("\n==============================")
print("KNN — INFO")
print("==============================")

info = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .select([
        pl.len().alias("total"),
        pl.col("Target").cast(pl.Float64).mean().alias("prop_pos"),
        pl.col("Anio").quantile(0.80).alias("anio_corte"),
    ])
    .collect()
)

total_filas = info["total"][0]
prop_pos    = info["prop_pos"][0]
anio_corte  = int(info["anio_corte"][0])
del info; gc.collect()

print(f"  Filas totales    : {total_filas:,}")
print(f"  Target = 1       : {prop_pos:.4f} ({prop_pos*100:.1f}%)")
print(f"  Corte train/test : <= {anio_corte} / > {anio_corte}")

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
        .filter(pl.col("Anio") <= anio_corte)
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


def sample_train(features, n_sample):
    train_info = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") <= anio_corte)
        .select([
            pl.len().alias("total"),
            pl.col("Target").cast(pl.Float64).mean().alias("prop"),
        ])
        .collect()
    )
    train_total = train_info["total"][0]
    train_prop  = train_info["prop"][0]
    del train_info

    n = min(n_sample, train_total)
    n_pos = int(train_prop * n)
    n_neg = n - n_pos
    print(f"  Muestra train: {n:,} (pos={n_pos:,}, neg={n_neg:,})")

    muestra_pos = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter((pl.col("Target") == 1) & (pl.col("Anio") <= anio_corte))
        .collect().sample(n=min(n_pos, train_total), seed=RANDOM_SEED)
    )
    muestra_neg = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter((pl.col("Target") == 0) & (pl.col("Anio") <= anio_corte))
        .collect().sample(n=min(n_neg, train_total), seed=RANDOM_SEED)
    )
    muestra = pl.concat([muestra_pos, muestra_neg]).sample(fraction=1.0, seed=RANDOM_SEED)
    del muestra_pos, muestra_neg; gc.collect()

    X = np.nan_to_num(muestra.select(features).to_numpy(allow_copy=True).astype(np.float32))
    y = muestra["Target"].to_numpy().astype(np.int8)
    del muestra; gc.collect()
    return X, y


def sample_test(features, n_sample):
    test_total = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") > anio_corte)
        .select(pl.len())
        .collect().item()
    )
    n = min(n_sample, test_total)
    muestra = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") > anio_corte)
        .collect().sample(n=n, seed=RANDOM_SEED)
    )
    X = np.nan_to_num(muestra.select(features).to_numpy(allow_copy=True).astype(np.float32))
    y = muestra["Target"].to_numpy().astype(np.int8)
    del muestra; gc.collect()
    return X, y


def evaluate_full_test(model, scaler, features, label):
    print(f"  [{label}] Evaluando en test completo...")
    y_true_all, y_proba_all = [], []

    for batch in (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") > anio_corte)
        .collect()
        .iter_slices(CHUNK_SIZE)
    ):
        X = np.nan_to_num(batch.select(features).to_numpy(allow_copy=True).astype(np.float32))
        y_true_all.extend(batch["Target"].to_numpy().astype(np.int8).tolist())
        y_proba_all.extend(model.predict_proba(scaler.transform(X))[:, 1].tolist())
        del X, batch; gc.collect()

    y_true  = np.array(y_true_all)
    y_proba = np.array(y_proba_all)
    y_pred  = (y_proba >= 0.5).astype(int)
    auc     = roc_auc_score(y_true, y_proba)

    print(f"\n  [{label}] ROC-AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))
    return {"label": label, "auc": auc, "y_true": y_true, "y_proba": y_proba, "y_pred": y_pred}

# =========================================================
# KNN: BARRIDO DE K + GRÁFICO DE CODO
# =========================================================

print("\n" + "="*55)
print("KNN — BARRIDO DE VECINOS")
print("="*55)

knn_all_results = {}

for feat_label, features in [("A", FEATURES_A), ("B", FEATURES_B)]:
    feat_desc = "Con NumOferentes" if feat_label == "A" else "Sin NumOferentes"
    print(f"\n--- Feature set {feat_label}: {feat_desc} ({len(features)} features) ---")

    scaler = build_scaler(features)
    X_train_raw, y_train = sample_train(features, N_MUESTRA)
    X_train = scaler.transform(X_train_raw)
    del X_train_raw; gc.collect()

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    del X_train, y_train; gc.collect()
    print(f"  Undersampling: {len(y_train_res):,} filas")

    N_TEST_KNN = 500_000
    X_test_raw, y_test = sample_test(features, N_TEST_KNN)
    X_test = scaler.transform(X_test_raw)
    del X_test_raw; gc.collect()
    print(f"  Muestra test (barrido): {len(y_test):,} filas")

    aucs = []
    for k in KNN_NEIGHBORS:
        print(f"  K={k:>3} ...", end=" ", flush=True)
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_res, y_train_res)
        y_proba = knn.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        aucs.append(auc)
        print(f"AUC={auc:.4f}")
        del knn, y_proba; gc.collect()

    knn_all_results[feat_label] = {
        "ks": KNN_NEIGHBORS, "aucs": aucs, "feat_desc": feat_desc,
        "best_k": KNN_NEIGHBORS[int(np.argmax(aucs))],
        "best_auc": max(aucs),
        "X_train_res": X_train_res, "y_train_res": y_train_res,
        "scaler": scaler, "features": features,
    }

    del X_test, y_test; gc.collect()

# Gráfico de codo
fig, ax = plt.subplots(figsize=(10, 6))
for feat_label, data in knn_all_results.items():
    ax.plot(data["ks"], data["aucs"], "o-",
            label=f"Set {feat_label}: {data['feat_desc']} (mejor K={data['best_k']})")
    best_idx = int(np.argmax(data["aucs"]))
    ax.annotate(f"K={data['best_k']}\nAUC={data['best_auc']:.4f}",
                xy=(data["ks"][best_idx], data["aucs"][best_idx]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlabel("K (número de vecinos)")
ax.set_ylabel("ROC-AUC")
ax.set_title("KNN — AUC vs K (gráfico de codo)")
ax.set_xticks(KNN_NEIGHBORS)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
guardar(fig, "modelo_KNN_codo.png")

# Evaluar mejor K en test completo
knn_final_results = []

for feat_label, data in knn_all_results.items():
    best_k = data["best_k"]
    label  = f"KNN-{feat_label}"
    print(f"\n  [{label}] Entrenando mejor K={best_k} para evaluación final...")

    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(data["X_train_res"], data["y_train_res"])

    result = evaluate_full_test(knn, data["scaler"], data["features"], label)
    knn_final_results.append(result)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(result["y_true"], result["y_pred"], ax=ax, colorbar=False)
    ax.set_title(f"{label} (K={best_k}) — Confusion Matrix (AUC={result['auc']:.4f})")
    guardar(fig, f"modelo_{label}_confusion.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(result["y_true"], result["y_proba"], ax=ax, name=f"{label} (K={best_k})")
    ax.set_title(f"{label} — Curva ROC (AUC={result['auc']:.4f})")
    guardar(fig, f"modelo_{label}_roc.png")

    del knn; gc.collect()

# Liberar datos KNN
for data in knn_all_results.values():
    del data["X_train_res"], data["y_train_res"]
del knn_all_results; gc.collect()

# =========================================================
# COMPARACIÓN + GUARDAR
# =========================================================

print("\n" + "="*55)
print("COMPARACIÓN KNN")
print("="*55)
print(f"  {'Modelo':<10} {'Features':<25} {'ROC-AUC':>10} {'Test filas':>12}")
print(f"  {'-'*60}")
for r in knn_final_results:
    feat_desc = "Con NumOferentes" if "A" in r["label"] else "Sin NumOferentes"
    print(f"  {r['label']:<10} {feat_desc:<25} {r['auc']:>10.4f} {len(r['y_true']):>12,}")

fig, ax = plt.subplots(figsize=(8, 7))
for r in knn_final_results:
    RocCurveDisplay.from_predictions(
        r["y_true"], r["y_proba"], ax=ax,
        name=f"{r['label']} (AUC={r['auc']:.4f})"
    )
ax.set_title("Comparación ROC — KNN")
ax.legend(loc="lower right")
guardar(fig, "modelo_KNN_comparacion_roc.png")

save_data = {}
save_labels = []
for r in knn_final_results:
    key = r["label"].replace("-", "_")
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
    save_data[f"{key}_fpr"] = fpr
    save_data[f"{key}_tpr"] = tpr
    save_data[f"{key}_auc"] = np.array([r["auc"]])
    cm = confusion_matrix(r["y_true"], r["y_pred"])
    save_data[f"{key}_cm"] = cm
    save_data[f"{key}_precision"] = np.array([precision_score(r["y_true"], r["y_pred"])])
    save_data[f"{key}_recall"] = np.array([recall_score(r["y_true"], r["y_pred"])])
    save_data[f"{key}_f1"] = np.array([f1_score(r["y_true"], r["y_pred"])])
    save_labels.append(r["label"])
save_data["labels"] = np.array(save_labels)
np.savez(os.path.join(RUTA_SALIDA, "resultados_knn.npz"), **save_data)
print("  Resultados guardados en resultados_knn.npz")

print(f"\nGráficos en: {RUTA_SALIDA}")
