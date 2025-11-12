import { Router } from "express";
import { z } from "zod";

import { correlations, linearRegression, normalize } from "../services/data";

const router = Router();

const normalizeSchema = z.object({
  values: z.array(z.number()).min(2, "provide at least two values"),
});

const regressionSchema = z.object({
  features: z
    .array(
      z.array(z.number()).nonempty("feature vectors cannot be empty"),
    )
    .min(2, "provide at least two observations"),
  targets: z.array(z.number()).min(2, "provide at least two targets"),
});

const correlateSchema = z.object({
  seriesA: z.array(z.number()).min(2, "seriesA requires at least two elements"),
  seriesB: z.array(z.number()).min(2, "seriesB requires at least two elements"),
});

router.post("/normalize", (req, res) => {
  const parsed = normalizeSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const result = normalize(parsed.data.values);
  return res.json({
    zscores: result.zScores,
    minMaxScaled: result.minMax,
    robustScaled: result.robust,
    statistics: result.stats,
  });
});

router.post("/regression", (req, res) => {
  const parsed = regressionSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const { features, targets } = parsed.data;
  try {
    const payload = linearRegression(features, targets);
    return res.json({
      coefficients: payload.coefficients,
      intercept: payload.intercept,
      rSquared: payload.rSquared,
      predictions: payload.predictions,
    });
  } catch (error) {
    return res.status(400).json({
      message: error instanceof Error ? error.message : "Regression failed",
    });
  }
});

router.post("/correlate", (req, res) => {
  const parsed = correlateSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  try {
    const payload = correlations(parsed.data.seriesA, parsed.data.seriesB);
    return res.json(payload);
  } catch (error) {
    return res.status(400).json({
      message: error instanceof Error ? error.message : "Correlation failed",
    });
  }
});

export default router;

