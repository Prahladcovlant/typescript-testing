import { Router } from "express";
import { z } from "zod";

import { summarizeText } from "../services/text";

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

export default router;

