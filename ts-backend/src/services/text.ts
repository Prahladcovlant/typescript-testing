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

export interface SummarizeResult {
  summary: string;
  sentences: string[];
  compressionRatio: number;
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

