"""
REST API FastAPI pour predictions High Financial Impact.

Composante 5 du sujet : containerise via Docker + endpoint /predict.

Endpoints :
- GET  /health   -> {status, model_loaded, model_name}
- GET  /         -> liens utiles
- POST /predict  -> retourne prediction binaire + probabilite

Le modele est charge depuis ../saved_models/best_model.joblib produit par
code/main.py. Si le fichier n'existe pas, l'API demarre quand meme mais
/predict retourne 503.

Lancement local :
    uvicorn code.api:app --reload --host 0.0.0.0 --port 8000
ou via Docker :
    docker build -t steve-ml-api .
    docker run -p 8000:8000 steve-ml-api
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "saved_models" / "best_model.joblib"

app = FastAPI(
    title="High Financial Impact predictor",
    description=(
        "Predit si un incident de cybersecurite aura un fort impact financier "
        "(>= quantile 0.75 de la distribution des pertes). "
        "Note : les modeles ont ete entraines sur le dataset Global Cybersecurity "
        "Threats 2015-2024 ; le diagnostic statistique a montre que ce dataset ne "
        "porte aucun signal exploitable. Les predictions ne sont pas operationnelles, "
        "l'API existe pour valider la chaine de deploiement."
    ),
    version="1.0.0",
)

_state = {"model": None, "model_name": None}


@app.on_event("startup")
def load_model():
    if MODEL_PATH.is_file():
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and "model" in bundle:
                _state["model"] = bundle["model"]
                _state["model_name"] = bundle.get("model_name", "unknown")
            else:
                _state["model"] = bundle
                _state["model_name"] = "unknown"
        except Exception as exc:
            print(f"[startup] failed to load model: {exc}")


class IncidentInput(BaseModel):
    Country: str = Field(..., examples=["Germany"])
    Year: int = Field(..., ge=2015, le=2030, examples=[2023])
    Attack_Type: str = Field(..., alias="Attack Type", examples=["DDoS"])
    Target_Industry: str = Field(..., alias="Target Industry", examples=["Banking"])
    Number_of_Affected_Users: int = Field(..., alias="Number of Affected Users", ge=0, examples=[150000])
    Attack_Source: str = Field(..., alias="Attack Source", examples=["Nation-state"])
    Security_Vulnerability_Type: str = Field(..., alias="Security Vulnerability Type", examples=["Zero-day"])
    Defense_Mechanism_Used: str = Field(..., alias="Defense Mechanism Used", examples=["Encryption"])
    Incident_Resolution_Time: int = Field(..., alias="Incident Resolution Time (in Hours)", ge=0, examples=[48])

    model_config = {"populate_by_name": True}


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    probability_high_impact: Optional[float] = None
    model_name: str
    note: str = (
        "Modele entraine sur dataset sans signal statistique exploitable -- "
        "predictions non operationnelles. Voir rapport diagnostic."
    )


@app.get("/")
def root():
    return {
        "service": "High Financial Impact predictor",
        "endpoints": ["/health", "/predict", "/docs", "/redoc"],
        "model_loaded": _state["model"] is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_name": _state["model_name"],
    }


def _add_engineered_features(df):
    """Reproduit add_feature_engineering_columns sans les loss-based features."""
    df = df.copy()
    eps = 1e-6
    if "Number of Affected Users" in df.columns and "Incident Resolution Time (in Hours)" in df.columns:
        df["Users Per Resolution Hour"] = df["Number of Affected Users"] / (
            df["Incident Resolution Time (in Hours)"].replace(0, None) + eps
        )
    if "Attack Source" in df.columns:
        df["Attack Source Is Unknown"] = (
            df["Attack Source"].astype(str).str.lower().eq("unknown").astype(int)
        )
    if "Year" in df.columns:
        df["Year Bucket"] = pd.cut(
            df["Year"],
            bins=[2014, 2018, 2021, 2024],
            labels=["2015-2018", "2019-2021", "2022-2024"],
            include_lowest=True,
        )
    return df


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: IncidentInput):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    record = payload.model_dump(by_alias=True)
    df = pd.DataFrame([record])
    df = _add_engineered_features(df)

    try:
        pred = int(_state["model"].predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")

    proba = None
    if hasattr(_state["model"], "predict_proba"):
        try:
            proba = float(_state["model"].predict_proba(df)[0, 1])
        except Exception:
            proba = None

    return PredictionOutput(
        prediction=pred,
        label="High Financial Impact" if pred == 1 else "Standard Impact",
        probability_high_impact=proba,
        model_name=_state["model_name"] or "unknown",
    )
