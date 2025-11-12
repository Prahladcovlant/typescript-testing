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

Additional endpoints from the legacy Python service will be ported incrementally.

# typescript-testing
