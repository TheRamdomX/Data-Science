import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

RUTA_PARQUET = r"C:\Users\matia\Desktop\DS\Datos_Limpios.parquet"
RUTA_SALIDA  = r"C:\Users\matia\Desktop\DS\EDA_Resultados"

import os
os.makedirs(RUTA_SALIDA, exist_ok=True)

columnas_utiles = [
    "Codigo", "Estado", "NombreOrganismo", "RegionUnidad", "sector",
    "Tipo de Adquisicion", "Tipo", "Moneda Adquisicion",
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos",
    "NombreProveedor", "Oferta seleccionada",
    "FechaCreacion", "FechaCierre", "FechaAdjudicacion",
    "UnidadTiempo", "TipoPago", "SubContratacion", "ComunaUnidad",
]

columnas_fecha = ["FechaCreacion", "FechaCierre", "FechaAdjudicacion"]
columnas_num   = ["MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos"]
columnas_cat   = ["Estado", "RegionUnidad", "sector", "Tipo de Adquisicion",
                  "Tipo", "TipoPago", "SubContratacion", "UnidadTiempo", "ComunaUnidad"]

# =========================================================
# SCAN BASE — función que reconstruye el lazy plan
# =========================================================
# CLAVE: base es una FUNCIÓN, no una variable
# Cada sección llama a get_base() → nuevo lazy plan desde disco
# Cuando termina la sección, el resultado (pequeño) se libera con gc.collect()
# Nunca hay más de una agregación en RAM al mismo tiempo
# =========================================================

def get_base():
    return (
        pl.scan_parquet(RUTA_PARQUET)
        .select(columnas_utiles)
        .with_columns([
            pl.when(pl.col(c) == pl.date(1900, 1, 1)).then(None).otherwise(pl.col(c)).alias(c)
            for c in columnas_fecha
        ])
        .with_columns([
            (pl.col("FechaAdjudicacion") - pl.col("FechaCreacion")).dt.total_days().alias("DiasAdjudicacion"),
            (pl.col("FechaCierre")       - pl.col("FechaCreacion")).dt.total_days().alias("DiasCierre"),
        ])
        .with_columns([
            (pl.col("Oferta seleccionada") == "Seleccionada").cast(pl.Int8).alias("Target"),
            pl.col("FechaCreacion").dt.year().alias("Anio"),
            pl.col("FechaCreacion").dt.month().alias("Mes"),
        ])
    )

schema_cols = get_base().collect_schema().names()

# =========================================================
# HELPERS
# =========================================================

def guardar(fig, nombre):
    ruta = os.path.join(RUTA_SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close(fig)
    gc.collect()  # liberar buffer matplotlib
    print(f"  Guardado: {ruta}")

def fmt_m(x, _):
    if abs(x) >= 1e9: return f"{x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"{x/1e3:.0f}K"
    return f"{x:.0f}"

def guardar_tabla(df_pl, nombre, titulo=""):
    df_plot = df_pl.head(30)
    alto = max(2, len(df_plot) * 0.4 + 1)
    fig, ax = plt.subplots(figsize=(max(10, len(df_pl.columns) * 1.5), alto))
    ax.axis("tight"); ax.axis("off")
    table = ax.table(cellText=df_plot.rows(), colLabels=df_plot.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.2, 1.4)
    if titulo: ax.set_title(titulo, weight="bold")
    guardar(fig, nombre)
    del df_plot; gc.collect()

# =========================================================
# SECCION 1 — INFO GENERAL
# =========================================================

print("\n=== 1. INFO GENERAL ===")

info = (
    get_base()
    .select(
        [pl.len().alias("__total__")]
        + [pl.col(c).is_null().sum().alias(f"null_{c}") for c in columnas_utiles
           if c in schema_cols]
    )
    .collect()
)
total = info["__total__"][0]
print(f"Filas: {total:,}")

cols_disponibles = [c for c in columnas_utiles if f"null_{c}" in info.columns]
resumen = pl.DataFrame({
    "Columna": cols_disponibles,
    "Missing": [info[f"null_{c}"][0] for c in cols_disponibles],
    "Pct (%)": [round(info[f"null_{c}"][0] / total * 100, 2) for c in cols_disponibles],
})
guardar_tabla(resumen, "01_missing.png", "Missing Values por Columna")
print(resumen)
del info, resumen; gc.collect()

# =========================================================
# SECCIONES 2-11
# =========================================================

def eda_estados():
    df = get_base().group_by("Estado").agg(pl.len().alias("Cantidad")).sort("Cantidad", descending=True).collect()
    guardar_tabla(df, "02_estados.png", "Estados de Licitación")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df["Estado"].to_list(), df["Cantidad"].to_list())
    ax.set_title("Distribución Estados"); plt.xticks(rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    guardar(fig, "02_estados_bar.png")
    del df; gc.collect()

def eda_regiones():
    df = get_base().group_by("RegionUnidad").agg(pl.len().alias("Cantidad")).sort("Cantidad", descending=True).collect()
    guardar_tabla(df, "03_regiones.png", "Regiones")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["RegionUnidad"].to_list(), df["Cantidad"].to_list())
    ax.set_title("Regiones"); plt.xticks(rotation=75, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    guardar(fig, "03_regiones_bar.png")
    del df; gc.collect()

def eda_sectores():
    df = (
        get_base()
        .filter(pl.col("sector").is_not_null() & ~pl.col("sector").is_in(["NA", "SINDATO", ""]))
        .group_by("sector").agg(pl.col("MontoEstimado").mean().alias("MontoPromedio"))
        .sort("MontoPromedio", descending=True).head(15).collect()
    )
    guardar_tabla(df, "04_sectores.png", "Monto Promedio por Sector")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["sector"].to_list(), df["MontoPromedio"].to_list())
    ax.set_title("Monto Promedio por Sector"); plt.xticks(rotation=75, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_m))
    guardar(fig, "04_sectores_bar.png")
    del df; gc.collect()

def eda_temporal():
    df = get_base().group_by("Anio").agg(pl.len().alias("Cantidad")).sort("Anio").collect()
    guardar_tabla(df, "09_temporal.png", "Licitaciones por Año")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Anio"].to_list(), df["Cantidad"].to_list(), marker="o")
    ax.set_title("Licitaciones por Año")
    guardar(fig, "09_temporal_line.png")
    del df; gc.collect()

def eda_target():
    df = get_base().group_by("Target").agg(pl.len().alias("Cantidad")).sort("Target").collect()
    guardar_tabla(df, "11_target.png", "Balance Target")
    ttl = df["Cantidad"].sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["No seleccionada (0)", "Seleccionada (1)"][:len(df)]
    ax.bar(labels, df["Cantidad"].to_list())
    for i, (c, p) in enumerate(zip(df["Cantidad"].to_list(), [c/ttl*100 for c in df["Cantidad"].to_list()])):
        ax.text(i, c, f"{p:.1f}%", ha="center", va="bottom")
    ax.set_title("Balance del Target")
    guardar(fig, "11_target_bar.png")
    del df; gc.collect()

for fn in [eda_estados, eda_regiones, eda_sectores, eda_temporal, eda_target]:
    fn()

# =========================================================
# SECCION 12 — PEARSON
# =========================================================

print("\n=== 12. CORRELACION PEARSON ===")

FEATURES_NUM = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos", "DiasAdjudicacion", "DiasCierre",
    "Anio", "Mes", "Target",
]

aggs = []
for c in FEATURES_NUM:
    aggs += [pl.col(c).mean().alias(f"mean_{c}"), pl.col(c).std(ddof=0).alias(f"std_{c}")]
    for c2 in FEATURES_NUM:
        aggs.append((pl.col(c) * pl.col(c2)).mean().alias(f"cov_{c}__{c2}"))

stats = (
    get_base()
    .select([pl.col(c).cast(pl.Float64, strict=False) for c in FEATURES_NUM])
    .select(aggs)
    .collect()
)

n = len(FEATURES_NUM)
corr_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        mi  = stats[f"mean_{FEATURES_NUM[i]}"][0] or 0.0
        mj  = stats[f"mean_{FEATURES_NUM[j]}"][0] or 0.0
        si  = stats[f"std_{FEATURES_NUM[i]}"][0]  or 0.0
        sj  = stats[f"std_{FEATURES_NUM[j]}"][0]  or 0.0
        exy = stats[f"cov_{FEATURES_NUM[i]}__{FEATURES_NUM[j]}"][0] or 0.0
        cov = exy - mi * mj
        den = si * sj
        val = cov / den if den > 1e-12 else 0.0
        corr_matrix[i][j] = max(min(val, 1.0), -1.0)

del stats; gc.collect()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(FEATURES_NUM, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(FEATURES_NUM, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr_matrix[i][j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(corr_matrix[i][j]) > 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Correlación de Pearson", fontsize=12)
fig.tight_layout()
guardar(fig, "12_pearson_extendido.png")
del corr_matrix; gc.collect()

# =========================================================
# SECCION 13 — CRAMÉR'S V
# =========================================================

print("\n=== 13. CRAMÉR'S V ===")

CATS = [c for c in ["Estado", "RegionUnidad", "sector", "Tipo de Adquisicion",
                     "Tipo", "TipoPago", "UnidadTiempo", "Anio", "Mes"] if c in schema_cols]

def cramers_v(col_a, col_b):
    try:
        ct = (
            get_base()
            .select([pl.col(col_a).cast(pl.Utf8), pl.col(col_b).cast(pl.Utf8)])
            .filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
            .group_by([col_a, col_b])
            .agg(pl.len().alias("obs"))
            .collect()
        )
        if ct.is_empty(): return 0.0
        rows = ct[col_a].unique().to_list()
        cols = ct[col_b].unique().to_list()
        if len(rows) < 2 or len(cols) < 2: return 0.0
        obs_dict = {(r[col_a], r[col_b]): r["obs"] for r in ct.iter_rows(named=True)}
        N = sum(obs_dict.values())
        row_totals = {r: sum(obs_dict.get((r, c), 0) for c in cols) for r in rows}
        col_totals = {c: sum(obs_dict.get((r, c), 0) for r in rows) for c in cols}
        chi2 = sum(
            (obs_dict.get((r, c), 0) - row_totals[r] * col_totals[c] / N) ** 2
            / (row_totals[r] * col_totals[c] / N)
            for r in rows for c in cols
            if row_totals[r] * col_totals[c] > 0
        )
        min_dim = min(len(rows), len(cols)) - 1
        v = (chi2 / (N * min_dim)) ** 0.5 if min_dim > 0 and N > 0 else 0.0
        del ct, obs_dict; gc.collect()
        return min(v, 1.0)
    except Exception as e:
        print(f"    Error ({col_a}, {col_b}): {e}"); return 0.0

m = len(CATS)
cv_matrix = np.zeros((m, m))
print(f"  Calculando {m*(m-1)//2} pares únicos Cramér's V...")
for i, ca in enumerate(CATS):
    for j, cb in enumerate(CATS):
        if i == j:   cv_matrix[i][j] = 1.0
        elif j > i:
            v = cramers_v(ca, cb)
            cv_matrix[i][j] = cv_matrix[j][i] = v
            print(f"    ({ca} x {cb}): {v:.3f}")

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(cv_matrix, vmin=0, vmax=1, cmap="YlOrRd")
ax.set_xticks(range(m)); ax.set_yticks(range(m))
ax.set_xticklabels(CATS, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(CATS, fontsize=9)
for i in range(m):
    for j in range(m):
        ax.text(j, i, f"{cv_matrix[i][j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if cv_matrix[i][j] > 0.7 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Cramér's V — Asociación entre Variables Categóricas", fontsize=12)
fig.tight_layout()
guardar(fig, "13_cramers_v.png")

pares = [{"Col_A": CATS[i], "Col_B": CATS[j], "CramersV": round(cv_matrix[i][j], 4)}
         for i in range(m) for j in range(i+1, m)]
pares_df = pl.DataFrame(pares).sort("CramersV", descending=True)
guardar_tabla(pares_df, "13b_cramers_pares.png", "Pares Categóricos por Cramér's V")
del cv_matrix, pares, pares_df; gc.collect()

# =========================================================
# SECCION 14 — ETA²
# =========================================================

print("\n=== 14. ETA² ===")

NUMS_ETA = ["MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "DiasAdjudicacion", "Target"]
CATS_ETA = [c for c in CATS if c in schema_cols]

def eta_squared(num_col, cat_col):
    try:
        glob = (
            get_base()
            .select([
                pl.col(num_col).cast(pl.Float64, strict=False).mean().alias("mean_global"),
                pl.col(num_col).cast(pl.Float64, strict=False).var(ddof=0).alias("var_total"),
                pl.col(num_col).cast(pl.Float64, strict=False).count().alias("N"),
            ])
            .collect()
        )
        mean_g = glob["mean_global"][0]
        var_t  = glob["var_total"][0]
        N      = glob["N"][0]
        if var_t is None or var_t < 1e-12 or N == 0: return 0.0
        grupos = (
            get_base()
            .filter(pl.col(cat_col).is_not_null() & pl.col(num_col).is_not_null())
            .group_by(cat_col)
            .agg([
                pl.col(num_col).cast(pl.Float64, strict=False).mean().alias("mean_g"),
                pl.col(num_col).cast(pl.Float64, strict=False).count().alias("n_g"),
            ])
            .collect()
        )
        if grupos.is_empty(): return 0.0
        ss_between = sum(
            r["n_g"] * (r["mean_g"] - mean_g) ** 2
            for r in grupos.iter_rows(named=True) if r["mean_g"] is not None
        )
        del glob, grupos; gc.collect()
        return min(max(ss_between / (var_t * N), 0.0), 1.0)
    except Exception as e:
        print(f"    Error eta² ({num_col}, {cat_col}): {e}"); return 0.0

rows_eta, cols_eta = len(NUMS_ETA), len(CATS_ETA)
eta_matrix = np.zeros((rows_eta, cols_eta))
for i, nc in enumerate(NUMS_ETA):
    for j, cc in enumerate(CATS_ETA):
        eta_matrix[i][j] = eta_squared(nc, cc)
        print(f"    ({nc} x {cc}): {eta_matrix[i][j]:.3f}")

fig, ax = plt.subplots(figsize=(max(10, cols_eta * 1.5), max(6, rows_eta)))
im = ax.imshow(eta_matrix, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(cols_eta)); ax.set_yticks(range(rows_eta))
ax.set_xticklabels(CATS_ETA, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(NUMS_ETA, fontsize=9)
for i in range(rows_eta):
    for j in range(cols_eta):
        ax.text(j, i, f"{eta_matrix[i][j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if eta_matrix[i][j] > 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Eta² — Variables Numéricas vs Categóricas", fontsize=12)
fig.tight_layout()
guardar(fig, "14_eta_squared.png")
del eta_matrix; gc.collect()

# =========================================================
# SECCION 15 — IMPACTO EN TARGET por variable categórica
# =========================================================

print("\n=== 15. TASA DE ADJUDICACIÓN POR CATEGORÍA ===")

for cat in ["Estado", "RegionUnidad", "sector", "Tipo de Adquisicion", "Tipo"]:
    if cat not in schema_cols:
        continue
    df = (
        get_base()
        .filter(pl.col(cat).is_not_null() & ~pl.col(cat).is_in(["NA", ""]))
        .group_by(cat)
        .agg([
            pl.len().alias("Total"),
            pl.col("Target").cast(pl.Float64).mean().alias("TasaAdjudicacion"),
        ])
        .sort("TasaAdjudicacion", descending=True)
        .head(20)
        .collect()
    )
    guardar_tabla(df, f"15_tasa_{cat.replace(' ','_')}.png", f"Tasa Adjudicación por {cat}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df[cat].to_list(), df["TasaAdjudicacion"].to_list())
    ax.set_title(f"Tasa de Adjudicación por {cat}")
    ax.set_ylabel("Tasa (0-1)"); ax.set_ylim(0, 1)
    plt.xticks(rotation=60, ha="right"); fig.tight_layout()
    guardar(fig, f"15_tasa_{cat.replace(' ','_')}_bar.png")
    del df; gc.collect()

# =========================================================
# SECCION 16 — MONTO ESTIMADO vs MONTO ADJUDICADO
# =========================================================

print("\n=== 16. RATIO MONTO ADJUDICADO / ESTIMADO ===")

ratio_df = (
    get_base()
    .filter(
        pl.col("MontoEstimado").is_not_null() & (pl.col("MontoEstimado") > 0) &
        pl.col("Valor Total Ofertado").is_not_null() & (pl.col("Valor Total Ofertado") > 0) &
        pl.col("sector").is_not_null() & ~pl.col("sector").is_in(["NA", "SINDATO", ""])
    )
    .with_columns([
        (pl.col("Valor Total Ofertado") / pl.col("MontoEstimado")).alias("Ratio")
    ])
    .filter(pl.col("Ratio").is_between(0, 10))
    .group_by("sector")
    .agg([
        pl.col("Ratio").mean().alias("RatioMedio"),
        pl.col("Ratio").median().alias("RatioMediana"),
        pl.len().alias("N"),
    ])
    .sort("RatioMedio", descending=True)
    .head(15)
    .collect()
)
guardar_tabla(ratio_df, "16_ratio_monto.png", "Ratio Adjudicado/Estimado por Sector")

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(ratio_df))
ax.bar(x, ratio_df["RatioMedio"].to_list(), label="Media")
ax.plot(x, ratio_df["RatioMediana"].to_list(), "ro-", label="Mediana")
ax.axhline(1.0, linestyle="--", color="gray", alpha=0.7, label="Ratio = 1 (exacto)")
ax.set_xticks(list(x)); ax.set_xticklabels(ratio_df["sector"].to_list(), rotation=60, ha="right")
ax.set_title("Ratio Monto Adjudicado / Monto Estimado por Sector")
ax.legend(); fig.tight_layout()
guardar(fig, "16_ratio_monto_bar.png")
del ratio_df; gc.collect()

# =========================================================
# SECCION 17 — ANÁLISIS DE COMPETENCIA (NumeroOferentes)
# =========================================================

print("\n=== 17. COMPETENCIA POR TIPO Y SECTOR ===")

for group_col in ["Tipo de Adquisicion", "sector"]:
    if group_col not in schema_cols:
        continue
    df = (
        get_base()
        .filter(
            pl.col(group_col).is_not_null() & ~pl.col(group_col).is_in(["NA", "SINDATO", ""]) &
            pl.col("NumeroOferentes").is_not_null()
        )
        .group_by(group_col)
        .agg([
            pl.col("NumeroOferentes").mean().alias("MediaOferentes"),
            pl.col("NumeroOferentes").median().alias("MedianaOferentes"),
            pl.len().alias("N"),
        ])
        .sort("MediaOferentes", descending=True)
        .head(15)
        .collect()
    )
    guardar_tabla(df, f"17_competencia_{group_col.replace(' ','_')}.png",
                  f"Competencia (Oferentes) por {group_col}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(df[group_col].to_list(), df["MediaOferentes"].to_list())
    ax.set_title(f"Promedio de Oferentes por {group_col}")
    ax.set_xlabel("Oferentes promedio"); fig.tight_layout()
    guardar(fig, f"17_competencia_{group_col.replace(' ','_')}_bar.png")
    del df; gc.collect()

# =========================================================
# SECCION 18 — RESUMEN ESTADÍSTICO NUMÉRICO
# =========================================================

print("\n=== 18. ESTADÍSTICAS DESCRIPTIVAS ===")

COLS_DESC = [
    "MontoEstimado", "Valor Total Ofertado", "NumeroOferentes",
    "CantidadReclamos", "DiasAdjudicacion", "DiasCierre",
]

aggs_desc = []
for c in COLS_DESC:
    aggs_desc += [
        pl.col(c).cast(pl.Float64, strict=False).count().alias(f"{c}__count"),
        pl.col(c).cast(pl.Float64, strict=False).mean().alias(f"{c}__mean"),
        pl.col(c).cast(pl.Float64, strict=False).std().alias(f"{c}__std"),
        pl.col(c).cast(pl.Float64, strict=False).min().alias(f"{c}__min"),
        pl.col(c).cast(pl.Float64, strict=False).quantile(0.25).alias(f"{c}__p25"),
        pl.col(c).cast(pl.Float64, strict=False).median().alias(f"{c}__median"),
        pl.col(c).cast(pl.Float64, strict=False).quantile(0.75).alias(f"{c}__p75"),
        pl.col(c).cast(pl.Float64, strict=False).max().alias(f"{c}__max"),
    ]

stats_desc = get_base().select(aggs_desc).collect()

stats_rows = []
for c in COLS_DESC:
    stats_rows.append({
        "Columna": c,
        "Count":   stats_desc[f"{c}__count"][0],
        "Mean":    round(stats_desc[f"{c}__mean"][0]   or 0, 2),
        "Std":     round(stats_desc[f"{c}__std"][0]    or 0, 2),
        "Min":     stats_desc[f"{c}__min"][0],
        "P25":     stats_desc[f"{c}__p25"][0],
        "Median":  stats_desc[f"{c}__median"][0],
        "P75":     stats_desc[f"{c}__p75"][0],
        "Max":     stats_desc[f"{c}__max"][0],
    })

desc_df = pl.DataFrame(stats_rows)
guardar_tabla(desc_df, "18_stats_descriptivas.png", "Estadísticas Descriptivas Numéricas")
print(desc_df)
del stats_desc, stats_rows, desc_df; gc.collect()

# =========================================================
# FIN
# =========================================================

print("\n" + "="*55)
print("EDA COMPLETO FINALIZADO")
print(f"Gráficos en: {RUTA_SALIDA}")
print("="*55)