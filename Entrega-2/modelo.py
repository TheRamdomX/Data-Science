import polars as pl
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import gc

# =========================================================
# CONFIG
# =========================================================

BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RUTA_LIMPIOS        = os.path.join(BASE_DIR, "Datos_Limpios.parquet")
RUTA_MODELO_PARQUET = os.path.join(BASE_DIR, "dataset_modelo.parquet")
RUTA_SALIDA         = os.path.join(BASE_DIR, "EDA_Resultados")
os.makedirs(RUTA_SALIDA, exist_ok=True)

RANDOM_SEED   = 42
CHUNK_SIZE    = 500_000
TEST_FRACTION = 0.20

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

ALL_FEATURES = list(dict.fromkeys(FEATURES_A + FEATURES_B))

# =========================================================
# FUNCIÓN BASE
# =========================================================

def get_base():
    return (
        pl.scan_parquet(RUTA_LIMPIOS)
        .with_columns([
            pl.when(pl.col(c) == pl.date(1900, 1, 1)).then(None).otherwise(pl.col(c)).alias(c)
            for c in ["FechaCreacion", "FechaCierre", "FechaAdjudicacion"]
        ])
        .with_columns([
            (pl.col("Oferta seleccionada") == "Seleccionada").cast(pl.Int8).alias("Target"),
            pl.col("FechaCreacion").dt.year().cast(pl.Int32).alias("Anio"),
        ])
    )

# =========================================================
# PASO 1 — CONSTRUIR dataset_modelo.parquet (si no existe)
# =========================================================

if os.path.exists(RUTA_MODELO_PARQUET):
    print(f"\n  dataset_modelo.parquet ya existe, saltando construcción")
else:
    print("\n==============================")
    print("1. DATASET DE MODELAMIENTO")
    print("==============================")

    FEATURES_NUM = [
        "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos",
    ]

    print("  Calculando medianas...")
    medianas_raw = (
        get_base()
        .select([pl.col(c).cast(pl.Float64, strict=False).median().alias(f"med_{c}") for c in FEATURES_NUM])
        .collect()
    )
    medianas = {c: (medianas_raw[f"med_{c}"][0] or 0.0) for c in FEATURES_NUM}
    print("  Medianas:", medianas)
    del medianas_raw; gc.collect()

    print("  Calculando frequency encoding...")
    total_n = get_base().select(pl.len()).collect().item()

    def freq_encoding_map(col_name):
        freq = (
            get_base()
            .filter(pl.col(col_name).is_not_null())
            .group_by(col_name)
            .agg(pl.len().alias("n"))
            .collect()
        )
        return {row[0]: row[1] / total_n for row in freq.rows()}

    map_region           = freq_encoding_map("RegionUnidad");        print("  RegionUnidad OK")
    map_sector           = freq_encoding_map("sector");              print("  sector OK")
    map_tipo_adquisicion = freq_encoding_map("Tipo de Adquisicion"); print("  Tipo de Adquisicion OK")
    gc.collect()

    print("  Escribiendo dataset_modelo.parquet...")
    (
        get_base()
        .with_columns([
            pl.col(c).cast(pl.Float64, strict=False).fill_null(medianas[c]).alias(c)
            for c in FEATURES_NUM
        ])
        .with_columns([
            pl.col("RegionUnidad").replace_strict(map_region,                  default=0.0).cast(pl.Float64).alias("fe_Region"),
            pl.col("sector").replace_strict(map_sector,                        default=0.0).cast(pl.Float64).alias("fe_Sector"),
            pl.col("Tipo de Adquisicion").replace_strict(map_tipo_adquisicion, default=0.0).cast(pl.Float64).alias("fe_TipoAdquisicion"),
        ])
        .select([*FEATURES_NUM, "fe_Region", "fe_Sector", "fe_TipoAdquisicion", "Anio", "Target"])
        .with_columns([
            pl.col(c).cast(pl.Float32, strict=False)
            for c in FEATURES_NUM + ["fe_Region", "fe_Sector", "fe_TipoAdquisicion"]
        ])
        .sink_parquet(RUTA_MODELO_PARQUET, compression="snappy")
    )

    check = pl.scan_parquet(RUTA_MODELO_PARQUET).select(pl.len()).collect().item()
    print(f"  Filas en dataset_modelo: {check:,}")
    del map_region, map_sector, map_tipo_adquisicion; gc.collect()

# =========================================================
# PASO 2 — INFO PREVIA + SPLIT
# =========================================================

print("\n==============================")
print("2. INFO DATASET MODELO")
print("==============================")

info = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .select([
        pl.len().alias("total"),
        pl.col("Target").cast(pl.Float64).mean().alias("prop_pos"),
        pl.col("Anio").quantile(1 - TEST_FRACTION).alias("anio_corte"),
    ])
    .collect()
)

total_filas = info["total"][0]
prop_pos    = info["prop_pos"][0]
prop_neg    = 1 - prop_pos
anio_corte  = int(info["anio_corte"][0])
del info; gc.collect()

print(f"  Filas totales         : {total_filas:,}")
print(f"  Target = 1 (pos)      : {prop_pos:.4f}  ({prop_pos*100:.1f}%)")
print(f"  Target = 0 (neg)      : {prop_neg:.4f}  ({prop_neg*100:.1f}%)")
print(f"  Corte train/test      : <= {anio_corte} / > {anio_corte}")

# =========================================================
# HELPERS
# =========================================================

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    gc.collect()
    print(f"  Guardado: {ruta}")

# =========================================================
# FUNCIONES DE ENTRENAMIENTO
# =========================================================

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
    scaler.mean_  = np.array([stats_sc[f"mean_{f}"][0] or 0.0            for f in features])
    scaler.scale_ = np.array([max(stats_sc[f"std_{f}"][0] or 1.0, 1e-8)  for f in features])
    scaler.n_features_in_ = len(features)
    del stats_sc; gc.collect()
    return scaler


def train_sgd(features, label):
    print(f"\n  [{label}] Entrenando SGDClassifier...")
    scaler = build_scaler(features)
    rus = RandomUnderSampler(random_state=RANDOM_SEED)

    model = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4,
        max_iter=1, random_state=RANDOM_SEED, warm_start=True, n_jobs=-1,
    )

    classes   = np.array([0, 1])
    n_batches = 0
    n_filas   = 0

    for batch in (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") <= anio_corte)
        .collect()
        .iter_slices(CHUNK_SIZE)
    ):
        X = np.nan_to_num(batch.select(features).to_numpy(allow_copy=True).astype(np.float32))
        y = batch["Target"].to_numpy().astype(np.int8)
        X_scaled = scaler.transform(X)
        if len(np.unique(y)) >= 2:
            X_res, y_res = rus.fit_resample(X_scaled, y)
        else:
            X_res, y_res = X_scaled, y
        model.partial_fit(X_res, y_res, classes=classes)
        n_batches += 1; n_filas += len(batch)
        if n_batches % 10 == 0:
            print(f"    Batch {n_batches} | {n_filas:,} filas")
        del X, y, X_scaled, X_res, y_res, batch; gc.collect()

    print(f"  [{label}] Entrenado: {n_filas:,} filas en {n_batches} batches")
    return model, scaler


def train_rf(features, label):
    print(f"\n  [{label}] Entrenando RandomForest...")
    scaler = build_scaler(features)

    N_MUESTRA = min(2_000_000, total_filas)
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

    n_sample = min(N_MUESTRA, train_total)
    n_pos = int(train_prop * n_sample)
    n_neg = n_sample - n_pos
    print(f"  [{label}] Muestra: {n_sample:,} (pos={n_pos:,}, neg={n_neg:,})")

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

    X_raw = np.nan_to_num(muestra.select(features).to_numpy(allow_copy=True).astype(np.float32))
    y_raw = muestra["Target"].to_numpy().astype(np.int8)
    del muestra; gc.collect()

    X_scaled = scaler.transform(X_raw)
    del X_raw; gc.collect()

    rus = RandomUnderSampler(random_state=RANDOM_SEED)
    X_res, y_res = rus.fit_resample(X_scaled, y_raw)
    print(f"  [{label}] Undersampling: {len(y_res):,} filas (pos={int(y_res.sum()):,})")
    del X_scaled, y_raw; gc.collect()

    model = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=30,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_res, y_res)
    del X_res, y_res; gc.collect()

    print(f"  [{label}] RandomForest entrenado")
    return model, scaler


def evaluate_model(model, scaler, features, label):
    print(f"  [{label}] Evaluando...")
    y_true_all, y_proba_all = [], []

    for batch in (
        pl.scan_parquet(RUTA_MODELO_PARQUET)
        .filter(pl.col("Anio") > anio_corte)
        .collect()
        .iter_slices(CHUNK_SIZE)
    ):
        X = np.nan_to_num(batch.select(features).to_numpy(allow_copy=True).astype(np.float32))
        y_true_all.extend(batch["Target"].to_numpy().astype(np.int8).tolist())
        X_scaled = scaler.transform(X)
        y_proba_all.extend(model.predict_proba(X_scaled)[:, 1].tolist())
        del X, X_scaled, batch; gc.collect()

    y_true  = np.array(y_true_all)
    y_proba = np.array(y_proba_all)
    y_pred  = (y_proba >= 0.5).astype(int)
    auc     = roc_auc_score(y_true, y_proba)

    print(f"\n  [{label}] ROC-AUC: {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    return {"label": label, "auc": auc, "y_true": y_true, "y_proba": y_proba, "y_pred": y_pred}


def plot_results(result, model, features, model_type):
    label = result["label"]
    auc   = result["auc"]

    if model_type == "sgd":
        coef = sorted(zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([f for f, _ in coef], [c for _, c in coef],
                color=["steelblue" if c > 0 else "tomato" for _, c in coef])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label} — Coeficientes (AUC={auc:.4f})")
        guardar(fig, f"modelo_{label}_coeficientes.png")
    else:
        importancias = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([f for f, _ in importancias], [i for _, i in importancias], color="steelblue")
        ax.set_title(f"{label} — Feature Importances (AUC={auc:.4f})")
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
# PASO 3 — ENTRENAR 4 VARIANTES
# =========================================================

print("\n" + "="*55)
print("3. ENTRENAMIENTO DE MODELOS")
print("="*55)

results = []

variants = [
    ("LR-A", "sgd", FEATURES_A),
    ("LR-B", "sgd", FEATURES_B),
    ("RF-A", "rf",  FEATURES_A),
    ("RF-B", "rf",  FEATURES_B),
]

for label, mtype, features in variants:
    print(f"\n{'='*55}")
    print(f"  VARIANTE: {label} ({mtype.upper()}, {len(features)} features)")
    print(f"{'='*55}")

    if mtype == "sgd":
        model, scaler = train_sgd(features, label)
    else:
        model, scaler = train_rf(features, label)

    result = evaluate_model(model, scaler, features, label)
    plot_results(result, model, features, mtype)
    results.append(result)

    del model, scaler; gc.collect()

# =========================================================
# PASO 4 — COMPARACIÓN + GUARDAR RESULTADOS
# =========================================================

print("\n" + "="*55)
print("4. COMPARACIÓN FINAL")
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
ax.set_title("Comparación ROC — LR + RF (undersampling)")
ax.legend(loc="lower right")
guardar(fig, "modelo_comparacion_roc.png")

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
np.savez(os.path.join(RUTA_SALIDA, "resultados_lr_rf.npz"), **save_data)
print("  Resultados guardados en resultados_lr_rf.npz")

print(f"\nGráficos en: {RUTA_SALIDA}")
