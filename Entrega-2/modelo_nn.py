import polars as pl
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
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
N_MUESTRA   = 2_000_000

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

# =========================================================
# INFO + SPLIT
# =========================================================

print("\n==============================")
print("RED NEURONAL (MLP) — INFO")
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


def evaluate(model, scaler, features, label):
    print(f"  [{label}] Evaluando en test...")
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
# ENTRENAR 2 VARIANTES
# =========================================================

results = []

for label, features in [("NN-A", FEATURES_A), ("NN-B", FEATURES_B)]:
    print(f"\n{'='*55}")
    print(f"  VARIANTE: {label} (MLPClassifier, {len(features)} features)")
    print(f"{'='*55}")

    scaler = build_scaler(features)
    X_raw, y_raw = sample_train(features, N_MUESTRA)
    X_scaled = scaler.transform(X_raw)
    del X_raw; gc.collect()

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X_scaled, y_raw)
    del X_scaled, y_raw; gc.collect()
    print(f"  [{label}] Undersampling: {len(y_res):,} filas (pos={int(y_res.sum()):,})")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=4096,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_SEED,
        verbose=True,
    )
    print(f"  [{label}] Entrenando MLP (128-64-32)...")
    model.fit(X_res, y_res)
    print(f"  [{label}] Épocas usadas: {model.n_iter_}")
    del X_res, y_res; gc.collect()

    result = evaluate(model, scaler, features, label)

    auc = result["auc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model.loss_curve_, label="Train loss", color="steelblue")
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, label="Val accuracy", color="tomato", linestyle="--")
        ax2.set_ylabel("Validation Accuracy")
        ax2.legend(loc="center right")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
    ax.set_title(f"{label} — Curva de Aprendizaje (AUC={auc:.4f})")
    ax.legend(loc="upper right")
    guardar(fig, f"modelo_{label}_learning_curve.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(result["y_true"], result["y_pred"], ax=ax, colorbar=False)
    ax.set_title(f"{label} — Confusion Matrix (AUC={auc:.4f})")
    guardar(fig, f"modelo_{label}_confusion.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(result["y_true"], result["y_proba"], ax=ax, name=label)
    ax.set_title(f"{label} — Curva ROC (AUC={auc:.4f})")
    guardar(fig, f"modelo_{label}_roc.png")

    results.append(result)
    del model, scaler; gc.collect()

# =========================================================
# COMPARACIÓN + GUARDAR
# =========================================================

print("\n" + "="*55)
print("COMPARACIÓN RED NEURONAL")
print("="*55)
print(f"  Split temporal: train <= {anio_corte} | test > {anio_corte}")
print(f"  {'Modelo':<10} {'Features':<25} {'ROC-AUC':>10} {'Test filas':>12}")
print(f"  {'-'*60}")
for r in results:
    feat_desc = "Con NumOferentes" if "A" in r["label"] else "Sin NumOferentes"
    print(f"  {r['label']:<10} {feat_desc:<25} {r['auc']:>10.4f} {len(r['y_true']):>12,}")

fig, ax = plt.subplots(figsize=(8, 7))
for r in results:
    RocCurveDisplay.from_predictions(
        r["y_true"], r["y_proba"], ax=ax,
        name=f"{r['label']} (AUC={r['auc']:.4f})"
    )
ax.set_title("Comparación ROC — Red Neuronal (MLP)")
ax.legend(loc="lower right")
guardar(fig, "modelo_NN_comparacion_roc.png")

save_data = {}
save_labels = []
for r in results:
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
np.savez(os.path.join(RUTA_SALIDA, "resultados_nn.npz"), **save_data)
print("  Resultados guardados en resultados_nn.npz")

print(f"\nGráficos en: {RUTA_SALIDA}")
