import polars as pl
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
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

# =========================================================
# INFO + SPLIT
# =========================================================

print("\n==============================")
print("GRADIENT BOOSTING — INFO")
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


def plot_results(result, model, scaler, features):
    label = result["label"]
    auc   = result["auc"]

    print(f"  [{label}] Calculando importancia por permutación...")
    test_sample = (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") > anio_corte)
        .collect()
        .sample(n=min(200_000, len(result["y_true"])), seed=RANDOM_SEED)
    )
    X_imp = scaler.transform(
        np.nan_to_num(test_sample.select(features).to_numpy(allow_copy=True).astype(np.float32))
    )
    y_imp = test_sample["Target"].to_numpy().astype(np.int8)
    del test_sample; gc.collect()

    perm = permutation_importance(model, X_imp, y_imp, n_repeats=5,
                                  scoring="roc_auc", random_state=RANDOM_SEED, n_jobs=-1)
    del X_imp, y_imp; gc.collect()

    importancias = sorted(zip(features, perm.importances_mean), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([f for f, _ in importancias], [i for _, i in importancias], color="darkgreen")
    ax.set_title(f"{label} — Permutation Importance (AUC={auc:.4f})")
    guardar(fig, f"modelo_{label}_importancias.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(result["y_true"], result["y_pred"], ax=ax, colorbar=False)
    ax.set_title(f"{label} — Confusion Matrix (AUC={auc:.4f})")
    guardar(fig, f"modelo_{label}_confusion.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(result["y_true"], result["y_proba"], ax=ax, name=label)
    ax.set_title(f"{label} — Curva ROC (AUC={auc:.4f})")
    guardar(fig, f"modelo_{label}_roc.png")

# =========================================================
# ENTRENAR 2 VARIANTES
# =========================================================

results = []

for label, features in [("GB-A", FEATURES_A), ("GB-B", FEATURES_B)]:
    print(f"\n{'='*55}")
    print(f"  VARIANTE: {label} (HistGradientBoosting, {len(features)} features)")
    print(f"{'='*55}")

    scaler = build_scaler(features)
    X_raw, y_raw = sample_train(features, N_MUESTRA)
    X_scaled = scaler.transform(X_raw)
    del X_raw; gc.collect()

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X_scaled, y_raw)
    del X_scaled, y_raw; gc.collect()
    print(f"  [{label}] Undersampling: {len(y_res):,} filas (pos={int(y_res.sum()):,})")

    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=8,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )
    print(f"  [{label}] Entrenando HistGradientBoosting...")
    model.fit(X_res, y_res)
    print(f"  [{label}] Iteraciones usadas: {model.n_iter_}")
    del X_res, y_res; gc.collect()

    result = evaluate(model, scaler, features, label)
    plot_results(result, model, scaler, features)
    results.append(result)
    del model, scaler; gc.collect()

# =========================================================
# COMPARACIÓN + GUARDAR
# =========================================================

print("\n" + "="*55)
print("COMPARACIÓN GRADIENT BOOSTING")
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
ax.set_title("Comparación ROC — Gradient Boosting")
ax.legend(loc="lower right")
guardar(fig, "modelo_GB_comparacion_roc.png")

save_data = {}
save_labels = []
for r in results:
    key = r["label"].replace("-", "_")
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"])
    save_data[f"{key}_fpr"] = fpr
    save_data[f"{key}_tpr"] = tpr
    save_data[f"{key}_auc"] = np.array([r["auc"]])
    save_labels.append(r["label"])
save_data["labels"] = np.array(save_labels)
np.savez(os.path.join(RUTA_SALIDA, "resultados_gb.npz"), **save_data)
print("  Resultados guardados en resultados_gb.npz")

print(f"\nGráficos en: {RUTA_SALIDA}")
