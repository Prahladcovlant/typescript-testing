# ToduFodu Backend

Power-packed FastAPI backend with 13 heavy-duty endpoints covering text intelligence, data science, finance, imaging, geospatial analytics, signal processing, task orchestration, and code analysis. Built so your frontend can hit the ground sprinting.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs live at `http://127.0.0.1:8000/docs`.

## Endpoints Snapshot

- `POST /text/summarize` – multi-factor extractive summarizer with compression metrics.
- `POST /text/sentiment` – lexicon-driven signal with polarity and dominant terms.
- `POST /text/keywords` – RAKE-inspired keyword and scoring engine.
- `POST /text/tfidf` – multi-document TF-IDF with configurable n-grams.
- `POST /data/normalize` – z-score and min-max scaling plus distribution stats.
- `POST /data/regression` – closed-form multivariate linear regression with R².
- `POST /data/correlate` – Pearson, Spearman, and Kendall rank correlations.
- `POST /finance/loan-amortization` – amortization schedule with early payoff support.
- `POST /image/invert` – base64 image inversion using Pillow.
- `POST /geo/distance` – geodesic Haversine distance in km and miles.
- `POST /signals/fourier` – discrete Fourier transform, dominant frequency, and energy.
- `POST /tasks/schedule` – topological scheduler with critical path extraction.
- `POST /code/dependencies` – AST-based dependency classifier (stdlib vs third-party).
- `POST /workflows/article-insights` – orchestrates summarization, sentiment, keywords, and TF-IDF over article batches.
- `POST /workflows/market-health` – fuses correlations, regression, and loan modeling to surface a portfolio stress score.
- `POST /workflows/project-blueprint` – combines scheduling, normalization, and signal analytics for project command centers.

Bring any frontend, plug into these endpoints, and unleash the todu-fodu workflows.

# python-testing
