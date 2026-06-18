import polars as pl
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gc

# =========================================================
# CONFIG
# =========================================================

RUTA_LIMPIOS        = r"C:\Users\matia\Desktop\DS\Datos_Limpios.parquet"
RUTA_MODELO_PARQUET = r"C:\Users\matia\Desktop\DS\dataset_modelo.parquet"
RUTA_SALIDA         = r"C:\Users\matia\Desktop\DS\EDA_Resultados"
os.makedirs(RUTA_SALIDA, exist_ok=True)

RANDOM_SEED   = 42
CHUNK_SIZE    = 500_000
TEST_FRACTION = 0.20

FEATURES_MODELO = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos", "DiasAdjudicacion", "DiasCierre",
    "Anio", "fe_Region", "fe_Sector", "fe_Estado",
]

# =========================================================
# FUNCIÓN BASE — igual que en EDA
# =========================================================
# Target y columnas derivadas se calculan aquí,
# no en el parquet limpio (que solo tiene datos crudos)
# =========================================================

def get_base():
    return (
        pl.scan_parquet(RUTA_LIMPIOS)
        .with_columns([
            pl.when(pl.col(c) == pl.date(1900, 1, 1)).then(None).otherwise(pl.col(c)).alias(c)
            for c in ["FechaCreacion", "FechaCierre", "FechaAdjudicacion"]
        ])
        # DiasAdjudicacion y DiasCierre ya están en el parquet limpio
        # solo necesitamos calcular Target y Anio
        .with_columns([
            (pl.col("Oferta seleccionada") == "Seleccionada").cast(pl.Int8).alias("Target"),
            pl.col("FechaCreacion").dt.year().cast(pl.Int32).alias("Anio"),
        ])
    )

# =========================================================
# PASO 1 — CONSTRUIR dataset_modelo.parquet
# =========================================================

print("\n==============================")
print("13. DATASET DE MODELAMIENTO")
print("==============================")

FEATURES_NUM = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos", "DiasAdjudicacion", "DiasCierre", "Anio",
]

# Medianas para imputación
print("  Calculando medianas...")
medianas_raw = (
    get_base()
    .select([pl.col(c).cast(pl.Float64, strict=False).median().alias(f"med_{c}") for c in FEATURES_NUM])
    .collect()
)
medianas = {c: (medianas_raw[f"med_{c}"][0] or 0.0) for c in FEATURES_NUM}
print("  Medianas:", medianas)
del medianas_raw; gc.collect()

# Frequency encoding
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

map_region = freq_encoding_map("RegionUnidad");  print("  RegionUnidad OK")
map_sector = freq_encoding_map("sector");         print("  sector OK")
map_estado = freq_encoding_map("Estado");         print("  Estado OK")
gc.collect()

# Escribir dataset_modelo en streaming
print("  Escribiendo dataset_modelo.parquet...")
(
    get_base()
    .with_columns([
        pl.col(c).cast(pl.Float64, strict=False).fill_null(medianas[c]).alias(c)
        for c in FEATURES_NUM
    ])
    .with_columns([
        pl.col("RegionUnidad").replace(map_region, default=0.0).cast(pl.Float64).alias("fe_Region"),
        pl.col("sector").replace(map_sector,       default=0.0).cast(pl.Float64).alias("fe_Sector"),
        pl.col("Estado").replace(map_estado,        default=0.0).cast(pl.Float64).alias("fe_Estado"),
    ])
    .select([*FEATURES_NUM, "fe_Region", "fe_Sector", "fe_Estado", "Target"])
    .with_columns([
        pl.col(c).cast(pl.Float32, strict=False)
        for c in FEATURES_NUM + ["fe_Region", "fe_Sector", "fe_Estado"]
    ])
    .sink_parquet(RUTA_MODELO_PARQUET, compression="snappy")
)

check = pl.scan_parquet(RUTA_MODELO_PARQUET).select(pl.len()).collect().item()
print(f"  Filas en dataset_modelo: {check:,}")
del map_region, map_sector, map_estado; gc.collect()

# =========================================================
# PASO 2 — INFO PREVIA
# =========================================================

print("\n==============================")
print("INFO DATASET MODELO")
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
# PASO 3 — LOGISTIC REGRESSION (incremental)
# =========================================================

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    gc.collect()
    print(f"  Guardado: {ruta}")

print("\n==============================")
print("14a. LOGISTIC REGRESSION")
print("==============================")

lr = SGDClassifier(
    loss="log_loss", penalty="l2", alpha=1e-4,
    max_iter=1, random_state=RANDOM_SEED, warm_start=True, n_jobs=-1,
)
scaler = StandardScaler()

# Estadísticos para scaler
stats_sc = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter(pl.col("Anio") <= anio_corte)
    .select([
        *[pl.col(f).cast(pl.Float64).mean().alias(f"mean_{f}") for f in FEATURES_MODELO],
        *[pl.col(f).cast(pl.Float64).std().alias(f"std_{f}")   for f in FEATURES_MODELO],
    ])
    .collect()
)
scaler.mean_  = np.array([stats_sc[f"mean_{f}"][0] or 0.0            for f in FEATURES_MODELO])
scaler.scale_ = np.array([max(stats_sc[f"std_{f}"][0] or 1.0, 1e-8)  for f in FEATURES_MODELO])
scaler.n_features_in_ = len(FEATURES_MODELO)
del stats_sc; gc.collect()

# Entrenamiento por chunks
print(f"  Entrenando (chunks de {CHUNK_SIZE:,}, train Anio <= {anio_corte})...")
classes    = np.array([0, 1])
n_batches  = 0
n_filas_lr = 0

for batch in (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter(pl.col("Anio") <= anio_corte)
    .collect()
    .iter_slices(CHUNK_SIZE)
):
    X = np.nan_to_num(batch.select(FEATURES_MODELO).to_numpy(allow_copy=True).astype(np.float32))
    y = batch["Target"].to_numpy().astype(np.int8)
    lr.partial_fit(scaler.transform(X), y, classes=classes)
    n_batches += 1; n_filas_lr += len(batch)
    if n_batches % 10 == 0:
        print(f"    Batch {n_batches} | {n_filas_lr:,} filas")
    del X, y, batch; gc.collect()

print(f"  LR entrenado: {n_filas_lr:,} filas en {n_batches} batches")

# Evaluación LR
y_true_lr, y_proba_lr = [], []
for batch in (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter(pl.col("Anio") > anio_corte)
    .collect()
    .iter_slices(CHUNK_SIZE)
):
    X = np.nan_to_num(batch.select(FEATURES_MODELO).to_numpy(allow_copy=True).astype(np.float32))
    y_true_lr.extend(batch["Target"].to_numpy().astype(np.int8).tolist())
    y_proba_lr.extend(lr.predict_proba(scaler.transform(X))[:, 1].tolist())
    del X, batch; gc.collect()

y_true_lr  = np.array(y_true_lr)
y_proba_lr = np.array(y_proba_lr)
y_pred_lr  = (y_proba_lr >= 0.5).astype(int)
auc_lr     = roc_auc_score(y_true_lr, y_proba_lr)
print(f"\n  ROC-AUC LR : {auc_lr:.4f}")
print(classification_report(y_true_lr, y_pred_lr, digits=4))

# Gráficos LR
coef_lr = sorted(zip(FEATURES_MODELO, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([f for f, _ in coef_lr], [c for _, c in coef_lr],
        color=["steelblue" if c > 0 else "tomato" for _, c in coef_lr])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"LR — Coeficientes (AUC={auc_lr:.4f})")
guardar(fig, "14a_lr_coeficientes.png")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_true_lr, y_pred_lr, ax=ax, colorbar=False)
ax.set_title(f"LR — Confusion Matrix (AUC={auc_lr:.4f})")
guardar(fig, "14a_lr_confusion.png")

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_true_lr, y_proba_lr, ax=ax, name="Logistic Regression")
ax.set_title(f"LR — Curva ROC (AUC={auc_lr:.4f})")
guardar(fig, "14a_lr_roc.png")

# =========================================================
# PASO 4 — RANDOM FOREST (muestra estratificada)
# =========================================================

print("\n==============================")
print("14b. RANDOM FOREST")
print("==============================")

N_MUESTRA_RF = min(10_000_000, total_filas)   
n_pos_rf = int(prop_pos * N_MUESTRA_RF)
n_neg_rf = N_MUESTRA_RF - n_pos_rf
print(f"  Muestra: {N_MUESTRA_RF:,} (pos={n_pos_rf:,}, neg={n_neg_rf:,})")

muestra_pos = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter((pl.col("Target") == 1) & (pl.col("Anio") <= anio_corte))
    .collect().sample(n=n_pos_rf, seed=RANDOM_SEED)
)
muestra_neg = (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter((pl.col("Target") == 0) & (pl.col("Anio") <= anio_corte))
    .collect().sample(n=n_neg_rf, seed=RANDOM_SEED)
)
muestra_rf = pl.concat([muestra_pos, muestra_neg]).sample(fraction=1.0, seed=RANDOM_SEED)
del muestra_pos, muestra_neg; gc.collect()

X_rf = np.nan_to_num(muestra_rf.select(FEATURES_MODELO).to_numpy(allow_copy=True).astype(np.float32))
y_rf = muestra_rf["Target"].to_numpy().astype(np.int8)
del muestra_rf; gc.collect()

rf = RandomForestClassifier(
    n_estimators=300, max_depth=14, min_samples_leaf=30,
    random_state=RANDOM_SEED, n_jobs=-1,
)
rf.fit(X_rf, y_rf)
del X_rf, y_rf; gc.collect()

# Evaluación RF
y_true_rf, y_proba_rf = [], []
for batch in (
    pl.scan_parquet(RUTA_MODELO_PARQUET)
    .filter(pl.col("Anio") > anio_corte)
    .collect()
    .iter_slices(CHUNK_SIZE)
):
    X = np.nan_to_num(batch.select(FEATURES_MODELO).to_numpy(allow_copy=True).astype(np.float32))
    y_true_rf.extend(batch["Target"].to_numpy().astype(np.int8).tolist())
    y_proba_rf.extend(rf.predict_proba(X)[:, 1].tolist())
    del X, batch; gc.collect()

y_true_rf  = np.array(y_true_rf)
y_proba_rf = np.array(y_proba_rf)
y_pred_rf  = (y_proba_rf >= 0.5).astype(int)
auc_rf     = roc_auc_score(y_true_rf, y_proba_rf)
print(f"\n  ROC-AUC RF : {auc_rf:.4f}")
print(classification_report(y_true_rf, y_pred_rf, digits=4))

# Gráficos RF
importancias = sorted(zip(FEATURES_MODELO, rf.feature_importances_), key=lambda x: x[1], reverse=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([f for f, _ in importancias], [i for _, i in importancias], color="steelblue")
ax.set_title(f"RF — Feature Importances (AUC={auc_rf:.4f})")
guardar(fig, "14b_rf_importancias.png")

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_true_rf, y_pred_rf, ax=ax, colorbar=False)
ax.set_title(f"RF — Confusion Matrix (AUC={auc_rf:.4f})")
guardar(fig, "14b_rf_confusion.png")

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_true_rf, y_proba_rf, ax=ax, name="Random Forest")
ax.set_title(f"RF — Curva ROC (AUC={auc_rf:.4f})")
guardar(fig, "14b_rf_roc.png")

# =========================================================
# COMPARACIÓN FINAL
# =========================================================

print("\n==============================")
print("RESUMEN MODELOS BASELINE")
print("==============================")
print(f"  Split temporal: train <= {anio_corte} | test > {anio_corte}")
print(f"  {'Modelo':<25} {'ROC-AUC':>10} {'Test filas':>15}")
print(f"  {'-'*52}")
print(f"  {'Logistic Regression':<25} {auc_lr:>10.4f} {len(y_true_lr):>15,}")
print(f"  {'Random Forest':<25} {auc_rf:>10.4f} {len(y_true_rf):>15,}")

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_true_lr, y_proba_lr, ax=ax, name=f"LR (AUC={auc_lr:.4f})")
RocCurveDisplay.from_predictions(y_true_rf, y_proba_rf, ax=ax, name=f"RF (AUC={auc_rf:.4f})")
ax.set_title("Comparación ROC — Modelos Baseline")
guardar(fig, "14_comparacion_roc.png")

print(f"\nGráficos en: {RUTA_SALIDA}")