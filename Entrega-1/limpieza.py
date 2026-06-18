import polars as pl

ruta_entrada = r"C:\Users\matia\Desktop\DS\Datos_Unificados.parquet"
ruta_salida  = r"C:\Users\matia\Desktop\DS\Datos_Limpios.parquet"

columnas_numericas = ["MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos"]
columnas_fecha     = ["FechaCreacion", "FechaCierre", "FechaAdjudicacion"]

# ── Calcular estadísticos solo sobre columnas numéricas (liviano) ─────────────
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

p999_monto_linea = stats["MontoLineaAdjudica"].drop_nulls().quantile(0.999)
del stats

# ── Pipeline lazy completo ────────────────────────────────────────────────────
df = pl.scan_parquet(ruta_entrada)

# Castear numéricas (siguen siendo String en el parquet unificado)
df = df.with_columns([
    pl.col(c).str.replace_all(r"\.", "").str.replace(",", ".").cast(pl.Float64, strict=False)
    for c in columnas_numericas
])

# Parsear fechas
df = df.with_columns([
    pl.col(c).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    for c in columnas_fecha
]).with_columns([
    pl.when(pl.col(c) == pl.date(1900, 1, 1)).then(None).otherwise(pl.col(c)).alias(c)
    for c in columnas_fecha
])

# Calcular DiasAdjudicacion y DiasCierre aquí, en limpieza
df = df.with_columns([
    (pl.col("FechaAdjudicacion") - pl.col("FechaCreacion")).dt.total_days().alias("DiasAdjudicacion"),
    (pl.col("FechaCierre")       - pl.col("FechaCreacion")).dt.total_days().alias("DiasCierre"),
])

# Eliminar negativos (incluye los días negativos)
df = df.filter(pl.all_horizontal([
    pl.col(c).is_null() | (pl.col(c) >= 0)
    for c in columnas_numericas + ["DiasAdjudicacion", "DiasCierre"]
]))

# Eliminar centinela MontoEstimado
df = df.filter(
    pl.col("MontoEstimado").is_null() | (pl.col("MontoEstimado") < 9_999_999_999_999_999)
)

# MontoLineaAdjudica por percentil 99.9
df = df.filter(
    pl.col("MontoLineaAdjudica").is_null() |
    pl.col("MontoLineaAdjudica").is_between(0, p999_monto_linea)
)

# IQR para el resto
for col, (lim_inf, lim_sup) in cuantiles.items():
    df = df.filter(
        pl.col(col).is_null() | pl.col(col).is_between(lim_inf, lim_sup)
    )

df.sink_parquet(ruta_salida)
print("Limpieza completa →", ruta_salida)

