import polars as pl

# Ruta del archivo Parquet generado en Data.py
ruta_parquet = r"C:\Users\matia\Desktop\DS\Datos_Unificados.parquet"

# Leer el archivo Parquet con Polars
df = pl.scan_parquet(ruta_parquet).collect()

#mostrar todas las columnas y su tipo de dato (lazy execution)
print(df.schema)

