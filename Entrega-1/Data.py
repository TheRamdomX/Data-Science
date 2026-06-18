import polars as pl
import os

ruta = r"C:\Users\matia\Desktop\DS\Datos"

ruta_salida = r"C:\Users\matia\Desktop\DS\Datos_Unificados.parquet"

archivos = [
    os.path.join(ruta, f)
    for f in os.listdir(ruta)
    if f.endswith(".csv")
]

# Lazy execution
df = pl.concat([
    pl.scan_csv(
        archivo,
        separator=";",
        encoding="utf8-lossy",
        ignore_errors=True,
        infer_schema_length=0  # forzar tipo de dato string
    )
    for archivo in archivos
], how="diagonal")  # Llenar de nulls las columnas faltantes

# Ejecutar y guardar en formato parquet 
df.sink_parquet(ruta_salida)

df_final = pl.scan_parquet(r"C:\Users\matia\Desktop\DS\Datos_Unificados.parquet")


print(df_final.head().collect())

cantidad_filas = df_final.select(pl.len()).collect().item()
print(f"Cantidad de filas exactas: {cantidad_filas}")

# Lista de columnas
columnas = df_final.collect_schema().names()
print(f"Columnas ({len(columnas)}): {columnas}")

resultado = (
    df_final
    .filter(pl.col("Estado") == "Adjudicada")
    .select(["Codigo", "NombreOrganismo", "MontoEstimado"])
    .collect() 
)

print(resultado)
