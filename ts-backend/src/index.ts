import cors from "cors";
import express, { NextFunction, Request, Response } from "express";

import dataRouter from "./routes/data";
import textRouter from "./routes/text";

const app = express();
const port = process.env.PORT ?? 4000;

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.get("/health", (_req: Request, res: Response) => {
  res.json({ status: "ok", service: "ts-backend" });
});

app.use("/text", textRouter);
app.use("/data", dataRouter);

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  // Basic error handler for now
  console.error(err);
  res.status(500).json({ message: "Internal Server Error" });
});

app.listen(port, () => {
  console.log(`TypeScript backend running on port ${port}`);
});

