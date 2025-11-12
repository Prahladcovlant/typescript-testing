const SENTENCE_SPLIT_REGEX = /(?<=[.!?])\s+/;
const WORD_REGEX = /[a-zA-Z']+/g;

const STOPWORDS = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "if",
  "while",
  "of",
  "to",
  "in",
  "on",
  "for",
  "with",
  "as",
  "is",
  "are",
  "was",
  "were",
  "be",
  "by",
  "at",
  "from",
  "that",
  "this",
  "it",
  "has",
  "have",
]);

const POSITIVE_TERMS = new Set([
  "great",
  "excellent",
  "amazing",
  "love",
  "happy",
  "superb",
  "fantastic",
  "positive",
  "success",
  "win",
  "delight",
  "enjoy",
  "progress",
  "growth",
]);

const NEGATIVE_TERMS = new Set([
  "bad",
  "terrible",
  "awful",
  "hate",
  "sad",
  "horrible",
  "negative",
  "fail",
  "loss",
  "problem",
  "issue",
  "down",
  "crash",
  "delay",
]);

export interface SummarizeResult {
  summary: string;
  sentences: string[];
  compressionRatio: number;
}

export interface SentimentResult {
  label: "positive" | "neutral" | "negative";
  polarity: number;
  positiveScore: number;
  negativeScore: number;
  neutralScore: number;
  topPositiveTerms: string[];
  topNegativeTerms: string[];
}

export interface KeywordResult {
  keywords: string[];
  scores: number[];
}

export interface TfIdfResult {
  vocabulary: string[];
  documentTopTerms: string[][];
}

export function summarizeText(text: string, maxSentences: number): SummarizeResult {
  const sentences = tokenizeSentences(text);
  if (sentences.length === 0 || maxSentences <= 0) {
    return { summary: "", sentences: [], compressionRatio: 0 };
  }

  const words = tokenizeWords(text);
  const frequency = computeFrequency(words);
  const normalizedFreq = normalizeFrequency(frequency);

  const sentenceScores = new Map<string, number>();
  sentences.forEach((sentence, index) => {
    const sentenceWords = tokenizeWords(sentence);
    let score = 0;
    sentenceWords.forEach((word) => {
      score += normalizedFreq.get(word) ?? 0;
    });
    const lengthPenalty = Math.log10(sentenceWords.length + 1);
    const positionBonus = 1 / (index + 1);
    const adjusted = score / (lengthPenalty || 1) + positionBonus;
    sentenceScores.set(sentence, adjusted);
  });

  const ranked = sentences
    .slice()
    .sort((a, b) => (sentenceScores.get(b) ?? 0) - (sentenceScores.get(a) ?? 0))
    .slice(0, Math.min(maxSentences, sentences.length))
    .sort((a, b) => sentences.indexOf(a) - sentences.indexOf(b));

  const summary = ranked.join(" ");
  const compressionRatio = summary.length / Math.max(text.length, 1);

  return {
    summary,
    sentences: ranked,
    compressionRatio: Number(compressionRatio.toFixed(4)),
  };
}

export function analyseSentiment(text: string): SentimentResult {
  const tokens = tokenizeWords(text);
  const occurrences = new Map<string, number>();
  tokens.forEach((token) => {
    occurrences.set(token, (occurrences.get(token) ?? 0) + 1);
  });

  let positive = 0;
  let negative = 0;
  occurrences.forEach((count, token) => {
    if (POSITIVE_TERMS.has(token)) positive += count;
    if (NEGATIVE_TERMS.has(token)) negative += count;
  });

  const total = positive + negative + tokens.length * 0.25;
  const positiveRatio = total ? positive / total : 0;
  const negativeRatio = total ? negative / total : 0;
  const neutralRatio = Math.max(0, 1 - (positiveRatio + negativeRatio));
  const polarity = positiveRatio - negativeRatio;

  const label: SentimentResult["label"] =
    polarity > 0.1 ? "positive" : polarity < -0.1 ? "negative" : "neutral";

  const topPositiveTerms = topTermsFromSet(occurrences, POSITIVE_TERMS);
  const topNegativeTerms = topTermsFromSet(occurrences, NEGATIVE_TERMS);

  return {
    label,
    polarity: Number(polarity.toFixed(4)),
    positiveScore: Number(positiveRatio.toFixed(4)),
    negativeScore: Number(negativeRatio.toFixed(4)),
    neutralScore: Number(neutralRatio.toFixed(4)),
    topPositiveTerms,
    topNegativeTerms,
  };
}

export function extractKeywords(text: string, topK: number): KeywordResult {
  const sentences = tokenizeSentences(text);
  const allWords = tokenizeWords(text);
  const frequency = computeFrequency(allWords);
  const degree = new Map<string, number>();

  sentences.forEach((sentence) => {
    const uniqueTerms = new Set(tokenizeWords(sentence));
    const sentenceDegree = uniqueTerms.size;
    uniqueTerms.forEach((term) => {
      degree.set(term, (degree.get(term) ?? 0) + sentenceDegree);
    });
  });

  const scores: Array<[string, number]> = [];
  degree.forEach((deg, term) => {
    const freq = frequency.get(term) ?? 1;
    scores.push([term, deg / freq]);
  });

  const ranked = scores
    .sort((a, b) => b[1] - a[1])
    .slice(0, Math.min(topK, scores.length));

  return {
    keywords: ranked.map(([term]) => term),
    scores: ranked.map(([, score]) => Number(score.toFixed(4))),
  };
}

export function computeTfIdf(
  documents: string[],
  topK: number,
  ngram: number,
): TfIdfResult {
  const tokenized = documents.map((doc) =>
    tokenizeWords(doc).filter((token) => token.length > 0),
  );
  const vocabCounter = new Map<string, number>();
  const docFrequency = new Map<string, number>();
  const docNgrams: string[][] = [];

  tokenized.forEach((tokens) => {
    const ngrams = generateNgrams(tokens, ngram);
    docNgrams.push(ngrams);
    const uniqueTerms = new Set(ngrams);
    uniqueTerms.forEach((term) =>
      docFrequency.set(term, (docFrequency.get(term) ?? 0) + 1),
    );
    ngrams.forEach((term) => {
      vocabCounter.set(term, (vocabCounter.get(term) ?? 0) + 1);
    });
  });

  const tfidfScores = docNgrams.map((ngrams) => {
    const counts = new Map<string, number>();
    ngrams.forEach((term) => counts.set(term, (counts.get(term) ?? 0) + 1));

    const docScores = new Map<string, number>();
    counts.forEach((count, term) => {
      const tf = count / ngrams.length;
      const df = docFrequency.get(term) ?? 1;
      const idf = Math.log((documents.length + 1) / (df + 1)) + 1;
      docScores.set(term, tf * idf);
    });
    return docScores;
  });

  const vocabulary = Array.from(vocabCounter.keys()).sort();

  const documentTopTerms = tfidfScores.map((docScores) =>
    Array.from(docScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, Math.min(topK, docScores.size))
      .map(([term]) => term),
  );

  return {
    vocabulary,
    documentTopTerms,
  };
}

function tokenizeSentences(text: string): string[] {
  return text
    .trim()
    .split(SENTENCE_SPLIT_REGEX)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function tokenizeWords(text: string): string[] {
  const matches = text.toLowerCase().match(WORD_REGEX);
  if (!matches) {
    return [];
  }
  return matches.filter((word) => !STOPWORDS.has(word) && word.length > 2);
}

function computeFrequency(words: string[]): Map<string, number> {
  const frequency = new Map<string, number>();
  words.forEach((word) => {
    frequency.set(word, (frequency.get(word) ?? 0) + 1);
  });
  return frequency;
}

function normalizeFrequency(freq: Map<string, number>): Map<string, number> {
  const values = Array.from(freq.values());
  const maxVal = Math.max(...values, 1);
  const normalized = new Map<string, number>();
  freq.forEach((value, key) => {
    normalized.set(key, value / maxVal);
  });
  return normalized;
}

function topTermsFromSet(
  occurrences: Map<string, number>,
  termSet: Set<string>,
  limit = 5,
): string[] {
  return Array.from(occurrences.entries())
    .filter(([term]) => termSet.has(term))
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
    .slice(0, limit)
    .map(([term]) => term);
}

function generateNgrams(tokens: string[], maxN: number): string[] {
  const results: string[] = [];
  const cappedN = Math.max(1, Math.min(maxN, 3));
  for (let n = 1; n <= cappedN; n += 1) {
    for (let i = 0; i <= tokens.length - n; i += 1) {
      results.push(tokens.slice(i, i + n).join(" "));
    }
  }
  return results;
}

