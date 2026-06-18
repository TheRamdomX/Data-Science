import polars as pl
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ruta_entrada  = os.path.join(BASE_DIR, "..", "Entrega-1", "Datos_Unificados.parquet")
ruta_salida   = os.path.join(BASE_DIR, "Datos_Limpios.parquet")

columnas_numericas = ["MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos"]
columnas_fecha     = ["FechaCreacion", "FechaCierre", "FechaAdjudicacion"]

# ── Calcular estadísticos solo sobre columnas numéricas ──────────────────────
stats = (
    pl.scan_parquet(ruta_entrada)
    .select(columnas_numericas)
    .with_columns([
        pl.col(c).str.replace_all(r"\.", "").str.replace(",", ".").cast(pl.Float64, strict=False)
        for c in columnas_numericas
    ])
    .collect()
)

cuantiles = {}
for col in ["MontoEstimado", "NumeroOferentes", "CantidadReclamos"]:
    s = stats[col].drop_nulls()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        cuantiles[col] = (q1 - 3.0 * iqr, q3 + 3.0 * iqr)

del stats

# MontoLineaAdjudica: computar p99.9 en scan separado (no está en columnas_numericas)
p999_monto_linea = (
    pl.scan_parquet(ruta_entrada)
    .select(
        pl.col("MontoLineaAdjudica")
        .str.replace_all(r"\.", "")
        .str.replace(",", ".")
        .cast(pl.Float64, strict=False)
        .alias("MontoLineaAdjudica")
    )
    .select(pl.col("MontoLineaAdjudica").drop_nulls().quantile(0.999))
    .collect()
    .item()
)

# ── Pipeline lazy completo ───────────────────────────────────────────────────
df = pl.scan_parquet(ruta_entrada)

df = df.with_columns([
    pl.col(c).str.replace_all(r"\.", "").str.replace(",", ".").cast(pl.Float64, strict=False)
    for c in columnas_numericas
])

# Castear MontoLineaAdjudica también
df = df.with_columns(
    pl.col("MontoLineaAdjudica")
    .str.replace_all(r"\.", "")
    .str.replace(",", ".")
    .cast(pl.Float64, strict=False)
    .alias("MontoLineaAdjudica")
)

df = df.with_columns([
    pl.col(c).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    for c in columnas_fecha
]).with_columns([
    pl.when(pl.col(c) == pl.date(1900, 1, 1)).then(None).otherwise(pl.col(c)).alias(c)
    for c in columnas_fecha
])

df = df.with_columns([
    (pl.col("FechaAdjudicacion") - pl.col("FechaCreacion")).dt.total_days().alias("DiasAdjudicacion"),
    (pl.col("FechaCierre")       - pl.col("FechaCreacion")).dt.total_days().alias("DiasCierre"),
])

df = df.filter(pl.all_horizontal([
    pl.col(c).is_null() | (pl.col(c) >= 0)
    for c in columnas_numericas + ["DiasAdjudicacion", "DiasCierre"]
]))

df = df.filter(
    pl.col("MontoEstimado").is_null() | (pl.col("MontoEstimado") < 9_999_999_999_999_999)
)

df = df.filter(
    pl.col("MontoLineaAdjudica").is_null() |
    pl.col("MontoLineaAdjudica").is_between(0, p999_monto_linea)
)

for col, (lim_inf, lim_sup) in cuantiles.items():
    df = df.filter(
        pl.col(col).is_null() | pl.col(col).is_between(lim_inf, lim_sup)
    )

df.sink_parquet(ruta_salida)
print("Limpieza completa →", ruta_salida)
