"""
FastAPI entrypoint for the multi-modal AKI prediction service.
Run with: uvicorn main:app --reload
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch

from schemas import PredictRequest, PredictResponse
from model import load_dummy_model
from utils import tokenize_text, clinical_to_tensor, timeseries_to_tensor

logger = logging.getLogger("aki_app")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AKI Prediction Service", version="1.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Model configuration
DEFAULT_CLINICAL_DIM = 4
DEFAULT_TS_DIM = 2
DEFAULT_VOCAB_SIZE = 10001

_model: Optional[torch.nn.Module] = None


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        logger.info("Loading AKI prediction model with dummy weights")
        _model = load_dummy_model(
            clinical_dim=DEFAULT_CLINICAL_DIM,
            timeseries_dim=DEFAULT_TS_DIM,
            vocab_size=DEFAULT_VOCAB_SIZE,
        )
    return _model


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    try:
        model = get_model()

        clinical = clinical_to_tensor(payload.clinical_features)
        timeseries = timeseries_to_tensor(payload.timeseries)
        # Infer timeseries dimension dynamically to accommodate variable input
        if timeseries.dim() != 3:
            raise ValueError("timeseries must be a 2D list of [time, features]")
        if clinical.size(1) != DEFAULT_CLINICAL_DIM:
            raise ValueError(
                f"Expected {DEFAULT_CLINICAL_DIM} clinical features, got {clinical.size(1)}"
            )
        if timeseries.size(2) != DEFAULT_TS_DIM:
            raise ValueError(
                f"Expected time-series feature dimension {DEFAULT_TS_DIM}, got {timeseries.size(2)}"
            )

        token_ids = tokenize_text(payload.note_text)
        text_tokens = token_ids.unsqueeze(0)

        with torch.no_grad():
            risk_tensor = model(clinical, timeseries, text_tokens)
            risk_score = float(risk_tensor.item())

        return PredictResponse(risk_score=risk_score)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):  # noqa: BLE001
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
