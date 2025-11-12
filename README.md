# TypeScript Backend

This project hosts a TypeScript/Express backend that is being built as the successor to the earlier Python FastAPI implementation.

## Getting Started

```bash
cd ts-backend
npm install
npm run dev         # start in watch mode
# or
npm run build
npm start
```

## Current Endpoints

- `GET /health` — service heartbeat.
- `POST /text/summarize` — extractive summarizer powered by the new TypeScript service layer.
- `POST /text/sentiment` — lexical sentiment signal with polarity + top positive/negative drivers.
- `POST /text/keywords` — RAKE-inspired keyword extraction with relevance scores.
- `POST /text/tfidf` — multi-document TF-IDF with configurable n-grams.
- `POST /data/normalize` — returns z-score, min-max, and robust scalings plus distribution stats.
- `POST /data/regression` — closed-form multivariate linear regression with R² and predictions.
- `POST /data/correlate` — Pearson, Spearman, and Kendall correlations for paired series.

# typescript-testing
