import json
import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "Data")

with open(os.path.join(MODELS_DIR, "meta.json")) as f:
    META = json.load(f)

bundle_05 = joblib.load(os.path.join(MODELS_DIR, "modelo_gb_t05.joblib"))
bundle_07 = joblib.load(os.path.join(MODELS_DIR, "modelo_gb_t07.joblib"))
bundle_09 = joblib.load(os.path.join(MODELS_DIR, "modelo_gb_t09.joblib"))

model = bundle_05["model"]
scaler = bundle_05["scaler"]
FEATURES = bundle_05["features"]

PROFILES = [
    {
        "key": "riesgoso",
        "threshold": 0.5,
        "name": "El Arriesgado",
        "avatar": "R",
        "description": "Threshold 0.5 — detecta mas oportunidades pero con mayor margen de error.",
        "style": "Optimista y audaz. Si hay chance, lo detecta.",
        "precision": bundle_05["metrics_at_threshold"]["precision"],
        "recall": bundle_05["metrics_at_threshold"]["recall"],
        "f1": bundle_05["metrics_at_threshold"]["f1"],
    },
    {
        "key": "moderado",
        "threshold": 0.7,
        "name": "El Equilibrado",
        "avatar": "M",
        "description": "Threshold 0.7 — equilibrio entre oportunidad y seguridad.",
        "style": "Prudente y analitico. Recomienda con buenas senales.",
        "precision": bundle_07["metrics_at_threshold"]["precision"],
        "recall": bundle_07["metrics_at_threshold"]["recall"],
        "f1": bundle_07["metrics_at_threshold"]["f1"],
    },
    {
        "key": "conservador",
        "threshold": 0.9,
        "name": "El Cauteloso",
        "avatar": "C",
        "description": "Threshold 0.9 — solo aprueba cuando esta muy seguro.",
        "style": "Riguroso y exigente. Minimiza falsos positivos.",
        "precision": bundle_09["metrics_at_threshold"]["precision"],
        "recall": bundle_09["metrics_at_threshold"]["recall"],
        "f1": bundle_09["metrics_at_threshold"]["f1"],
    },
]

del bundle_05, bundle_07, bundle_09


def _load_csv():
    frames = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(
            os.path.join(DATA_DIR, fname),
            sep=";",
            encoding="latin-1",
            dtype=str,
            on_bad_lines="skip",
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    keep = [
        "CodigoExterno", "Nombre", "MontoEstimado", "Valor Total Ofertado",
        "NumeroOferentes", "CantidadReclamos",
        "RegionUnidad", "sector", "Tipo de Adquisicion",
        "NombreProveedor", "Oferta seleccionada",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    for col in ["MontoEstimado", "Valor Total Ofertado", "NumeroOferentes", "CantidadReclamos"]:
        if col in df.columns:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["MontoEstimado", "Valor Total Ofertado", "RegionUnidad", "sector", "Tipo de Adquisicion"])
    df["NumeroOferentes"] = df["NumeroOferentes"].fillna(0).astype(int)
    df["CantidadReclamos"] = df["CantidadReclamos"].fillna(0).astype(int)

    valid_regions = set(META["categories"]["RegionUnidad"])
    valid_sectors = set(META["categories"]["sector"])
    valid_tipos = set(META["categories"]["Tipo de Adquisicion"])
    df = df[
        df["RegionUnidad"].isin(valid_regions)
        & df["sector"].isin(valid_sectors)
        & df["Tipo de Adquisicion"].isin(valid_tipos)
    ].reset_index(drop=True)

    return df


print("Cargando datos CSV...")
csv_data = _load_csv()
print(f"CSV cargado: {len(csv_data)} filas validas")


def _find_optimal_value(monto_est, n_oferentes, reclamos, region, sector, tipo, n_steps=60):
    freq_maps = META["freq_maps"]
    fe_region = freq_maps["RegionUnidad"].get(region, 0.0)
    fe_sector = freq_maps["sector"].get(sector, 0.0)
    fe_tipo = freq_maps["Tipo de Adquisicion"].get(tipo, 0.0)

    low = max(monto_est * 0.05, 1000)
    high = monto_est * 3.0
    values = np.linspace(low, high, n_steps)

    X = np.array([
        [monto_est, v, n_oferentes, reclamos, fe_region, fe_sector, fe_tipo]
        for v in values
    ], dtype=np.float32)
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[:, 1]

    best_idx = int(np.argmax(probas))
    return float(values[best_idx]), float(probas[best_idx])


def _build_desiertas_index(df):
    if "CodigoExterno" not in df.columns:
        return []
    has_sel = df.loc[df["Oferta seleccionada"] == "Seleccionada", "CodigoExterno"].unique()
    has_sel_set = set(has_sel)
    desiertas_df = df[~df["CodigoExterno"].isin(has_sel_set)]
    grouped = desiertas_df.groupby("CodigoExterno", sort=False)
    items = []
    for code, group in grouped:
        row = group.iloc[0]
        items.append({
            "codigo": str(code),
            "nombre": str(row.get("Nombre", "")) if pd.notna(row.get("Nombre")) else "",
            "monto_estimado": float(row["MontoEstimado"]),
            "numero_oferentes": int(row["NumeroOferentes"]),
            "cantidad_reclamos": int(row["CantidadReclamos"]),
            "region": str(row["RegionUnidad"]),
            "sector": str(row["sector"]),
            "tipo_adquisicion": str(row["Tipo de Adquisicion"]),
            "n_ofertas": len(group),
        })
    return items


print("Indexando licitaciones desiertas...")
desiertas_list = _build_desiertas_index(csv_data)
print(f"Desiertas: {len(desiertas_list)} licitaciones sin adjudicar")

app = FastAPI(title="Predictor de Licitaciones")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LicitacionInput(BaseModel):
    monto_estimado: float
    valor_total_ofertado: float
    numero_oferentes: int
    cantidad_reclamos: int
    region: str
    sector: str
    tipo_adquisicion: str


def build_features(data: LicitacionInput) -> np.ndarray:
    freq_maps = META["freq_maps"]
    fe_region = freq_maps["RegionUnidad"].get(data.region, 0.0)
    fe_sector = freq_maps["sector"].get(data.sector, 0.0)
    fe_tipo = freq_maps["Tipo de Adquisicion"].get(data.tipo_adquisicion, 0.0)

    x = np.array([[
        data.monto_estimado,
        data.valor_total_ofertado,
        data.numero_oferentes,
        data.cantidad_reclamos,
        fe_region,
        fe_sector,
        fe_tipo,
    ]], dtype=np.float32)

    return scaler.transform(x)


def _build_recommendations(data: LicitacionInput, probability: float):
    medianas = META["medianas"]
    comparisons = []

    def _cmp(field_label, value, median):
        if median == 0:
            return
        ratio = value / median
        if ratio > 1.5:
            status = "high"
        elif ratio < 0.5:
            status = "low"
        else:
            status = "normal"
        comparisons.append({
            "field": field_label,
            "value": value,
            "median": median,
            "ratio": round(ratio, 2),
            "status": status,
        })

    _cmp("Monto Estimado", data.monto_estimado, medianas["MontoEstimado"])
    _cmp("Valor Ofertado", data.valor_total_ofertado, medianas["Valor Total Ofertado"])
    _cmp("N Oferentes", data.numero_oferentes, medianas["NumeroOferentes"])
    _cmp("Reclamos", data.cantidad_reclamos, medianas["CantidadReclamos"])

    tips = []
    if probability < 30:
        tips = [
            "La probabilidad de seleccion es baja. Considere revisar la estrategia de oferta.",
            "Evalue si el valor ofertado es competitivo para esta region y sector.",
            "Un alto numero de reclamos puede estar afectando negativamente.",
        ]
    elif probability < 50:
        tips = [
            "Posibilidades moderadas. La competencia puede ser un factor clave.",
            "Revise si los terminos se ajustan al perfil tipico de licitaciones exitosas en este sector.",
        ]
    elif probability < 70:
        tips = [
            "Buenas senales. El perfil se acerca al de ofertas seleccionadas.",
            "Mantenga un valor competitivo y asegurese de cumplir requisitos formales.",
        ]
    else:
        tips = [
            "Excelentes perspectivas. El perfil coincide con ofertas seleccionadas historicamente.",
            "Asegurese de mantener la calidad tecnica y cumplimiento de plazos.",
        ]

    return {"tips": tips, "comparisons": comparisons}


@app.get("/options")
def get_options():
    return {
        "regiones": META["categories"]["RegionUnidad"],
        "sectores": META["categories"]["sector"],
        "tipos_adquisicion": META["categories"]["Tipo de Adquisicion"],
    }


@app.post("/predict")
def predict(data: LicitacionInput):
    X = build_features(data)
    proba = float(model.predict_proba(X)[0, 1])

    results = []
    for profile in PROFILES:
        threshold = profile["threshold"]
        selected = proba >= threshold

        if selected:
            verdict = "Recomendada para postular"
        elif proba >= threshold * 0.7:
            verdict = "En zona gris - evaluar con cuidado"
        else:
            verdict = "No recomendada segun este perfil"

        results.append({
            "key": profile["key"],
            "name": profile["name"],
            "avatar": profile["avatar"],
            "description": profile["description"],
            "style": profile["style"],
            "threshold": threshold,
            "probability": round(proba * 100, 1),
            "verdict": verdict,
            "selected": selected,
            "model_precision": round(profile["precision"] * 100, 1),
            "model_recall": round(profile["recall"] * 100, 1),
            "model_f1": round(profile["f1"] * 100, 1),
        })

    recommendations = _build_recommendations(data, proba * 100)

    return {
        "probability": round(proba * 100, 1),
        "predictions": results,
        "recommendations": recommendations,
    }


def _row_to_dict(row):
    return {
        "nombre": str(row.get("Nombre", "")) if pd.notna(row.get("Nombre")) else "",
        "monto_estimado": float(row["MontoEstimado"]),
        "valor_total_ofertado": float(row["Valor Total Ofertado"]),
        "numero_oferentes": int(row["NumeroOferentes"]),
        "cantidad_reclamos": int(row["CantidadReclamos"]),
        "region": str(row["RegionUnidad"]),
        "sector": str(row["sector"]),
        "tipo_adquisicion": str(row["Tipo de Adquisicion"]),
        "proveedor": str(row.get("NombreProveedor", "")) if pd.notna(row.get("NombreProveedor")) else "",
        "resultado_real": str(row.get("Oferta seleccionada", "")) if pd.notna(row.get("Oferta seleccionada")) else "",
    }


@app.get("/csv-data")
def get_csv_data(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    total = len(csv_data)
    start = (page - 1) * size
    end = min(start + size, total)
    if start >= total:
        return {"total": total, "page": page, "size": size, "data": []}
    records = [_row_to_dict(csv_data.iloc[i]) for i in range(start, end)]
    return {"total": total, "page": page, "size": size, "data": records}


@app.get("/csv-random")
def get_csv_random():
    if csv_data.empty:
        return {"error": "No hay datos disponibles"}
    row = csv_data.sample(n=1).iloc[0]
    return _row_to_dict(row)


@app.get("/desiertas")
def get_desiertas(page: int = Query(1, ge=1), size: int = Query(15, ge=1, le=50)):
    total = len(desiertas_list)
    start = (page - 1) * size
    end = min(start + size, total)
    if start >= total:
        return {"total": total, "page": page, "size": size, "data": []}

    results = []
    for item in desiertas_list[start:end]:
        val_sug, prob_sug = _find_optimal_value(
            item["monto_estimado"], item["numero_oferentes"],
            item["cantidad_reclamos"], item["region"],
            item["sector"], item["tipo_adquisicion"],
        )
        results.append({
            **item,
            "valor_sugerido": round(val_sug, 0),
            "prob_sugerida": round(prob_sug * 100, 1),
            "aprueba_riesgoso": prob_sug >= 0.5,
            "aprueba_moderado": prob_sug >= 0.7,
            "aprueba_conservador": prob_sug >= 0.9,
        })
    return {"total": total, "page": page, "size": size, "data": results}
