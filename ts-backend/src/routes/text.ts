import { Router } from "express";
import { z } from "zod";

import {
  analyseSentiment,
  buildTextInsights,
  computeTfIdf,
  extractKeywords,
  summarizeText,
} from "../services/text";

const router = Router();

const summarizeSchema = z.object({
  text: z.string().min(10, "text must be at least 10 characters"),
  maxSentences: z.coerce
    .number()
    .int("maxSentences must be an integer")
    .min(1, "maxSentences must be at least 1")
    .max(10, "maxSentences must be at most 10")
    .default(3),
});

const sentimentSchema = z.object({
  text: z.string().min(3, "text must be at least 3 characters"),
});

const keywordsSchema = z.object({
  text: z.string().min(10, "text must be at least 10 characters"),
  topK: z.coerce
    .number()
    .int("topK must be an integer")
    .min(1, "topK must be at least 1")
    .max(15, "topK cannot exceed 15")
    .default(5),
});

const tfidfSchema = z.object({
  documents: z
    .array(z.string().min(5, "each document must be at least 5 characters"))
    .min(2, "provide at least two documents"),
  topK: z.coerce
    .number()
    .int("topK must be an integer")
    .min(1, "topK must be at least 1")
    .max(20, "topK cannot exceed 20")
    .default(5),
  ngram: z.coerce
    .number()
    .int("ngram must be an integer")
    .min(1, "ngram must be at least 1")
    .max(3, "ngram cannot exceed 3")
    .default(1),
});

const insightsSchema = z.object({
  text: z.string().min(10, "text must be at least 10 characters"),
  maxSentences: z.coerce
    .number()
    .int("maxSentences must be an integer")
    .min(1, "maxSentences must be at least 1")
    .max(10, "maxSentences must be at most 10")
    .default(3),
  keywordTopK: z.coerce
    .number()
    .int("keywordTopK must be an integer")
    .min(1, "keywordTopK must be at least 1")
    .max(20, "keywordTopK cannot exceed 20")
    .default(5),
});

router.post("/summarize", (req, res) => {
  const parseResult = summarizeSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parseResult.error.flatten(),
    });
  }

  const { text, maxSentences } = parseResult.data;
  const summary = summarizeText(text, maxSentences);
  return res.json({
    summary: summary.summary,
    sentences: summary.sentences,
    compressionRatio: summary.compressionRatio,
  });
});

router.post("/sentiment", (req, res) => {
  const parsed = sentimentSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const payload = analyseSentiment(parsed.data.text);
  return res.json({
    label: payload.label,
    polarity: payload.polarity,
    positiveScore: payload.positiveScore,
    negativeScore: payload.negativeScore,
    neutralScore: payload.neutralScore,
    topPositiveTerms: payload.topPositiveTerms,
    topNegativeTerms: payload.topNegativeTerms,
  });
});

router.post("/keywords", (req, res) => {
  const parsed = keywordsSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const payload = extractKeywords(parsed.data.text, parsed.data.topK);
  return res.json(payload);
});

router.post("/tfidf", (req, res) => {
  const parsed = tfidfSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const { documents, topK, ngram } = parsed.data;
  const payload = computeTfIdf(documents, topK, ngram);
  return res.json(payload);
});

router.post("/insights", (req, res) => {
  const parsed = insightsSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(422).json({
      message: "Validation failed",
      errors: parsed.error.flatten(),
    });
  }

  const { text, maxSentences, keywordTopK } = parsed.data;
  const payload = buildTextInsights(text, maxSentences, keywordTopK);
  return res.json({
    summary: payload.summary,
    sentiment: payload.sentiment,
    keywords: payload.keywords,
  });
});

export default router;

