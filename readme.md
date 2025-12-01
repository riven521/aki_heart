You are an expert Python full-stack developer and medical-AI system architect.
Generate a fully runnable **FastAPI + PyTorch + Web front-end** multi-modal AKI prediction system.
The output must include ALL files with complete runnable code (no placeholders, no omissions).

────────────────────────────────────────
【Project Goal】
Build the complete implementation of:
“基于多模态异构数据的心脏外科术后急性肾损伤风险预测系统 V1.0”

The system must include:
1. A FastAPI backend for model inference.
2. A PyTorch multi-modal deep learning model (structured + text + time-series).
3. A web-based user interface (HTML + JavaScript).
4. A full working pipeline: upload → predict → display results.

────────────────────────────────────────
【Tech Stack】
Backend:
- Python 3.10+
- FastAPI
- PyTorch
- transformers (for text encoder, or DummyEncoder)
- Pydantic
- Uvicorn
- Jinja2 templates for HTML rendering

Frontend:
- HTML + CSS + JavaScript (no frameworks required)
- Fetch API to call backend predict endpoint
- SimpleBootstrap UI or pure HTML

────────────────────────────────────────
【Functional Requirements】

Backend API:
- POST /predict
- Input JSON:
  {
    "clinical_features": List[float],
    "note_text": "string",
    "timeseries": List[List[float]]
  }
- Output JSON:
  { "risk_score": float }

Deep Learning Model:
- A PyTorch module combining:
  - MLP for clinical features
  - LSTM for time-series
  - BERT encoder or DummyTextEncoder for text
  - Concat or attention fusion
- Must load successfully and run forward() without errors
- If real weights not available, generate dummy random weights

Web UI:
- Route GET / → render a web page
- Page contains:
  - Text input box for clinical variables
  - Textarea for note_text
  - Upload field or multiline field for timeseries
  - A “Predict Risk” button
  - JavaScript fetch() call to /predict
  - Result display block that shows:
      • risk score (0–1)
      • risk category (low / medium / high)
- Display errors gracefully

────────────────────────────────────────
【Required File Structure — Codex MUST output all files】

project/
│── main.py             (FastAPI app, HTML template rendering, /predict)
│── model.py            (PyTorch multi-modal model)
│── schemas.py          (Pydantic input/output schemas)
│── utils.py            (preprocessing helpers)
│── requirements.txt    (auto-generate minimal working list)
│
├── templates/
│     └── index.html    (full HTML page with JS calling /predict)
│
└── static/
      └── style.css     (basic styling)

────────────────────────────────────────
【Quality Requirements】
- ALL code must be complete and runnable.
- No placeholders, no “...” allowed.
- Must simulate a working PyTorch model if no weights exist.
- Provide full HTML + JS (including fetch() to /predict).
- Include uvicorn run instruction:  uvicorn main:app --reload
- Include detailed error handling both frontend & backend.

────────────────────────────────────────
【Example Input】
{
  "clinical_features": [85.0, 1.2, 3.5, 0.9],
  "note_text": "post-operative urine output decreased...",
  "timeseries": [[80,120],[78,115],[75,118]]
}

【Example Output】
{ "risk_score": 0.37 }

────────────────────────────────────────
Please output ALL source files now with complete runnable code.
