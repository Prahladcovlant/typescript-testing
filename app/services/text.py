from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

STOPWORDS = {
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
}

POSITIVE_TERMS = {
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
}

NEGATIVE_TERMS = {
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
}


def sentence_tokenize(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def summarize(text: str, max_sentences: int) -> Tuple[str, List[str], float]:
    sentences = sentence_tokenize(text)
    if not sentences:
        return "", [], 0.0

    words = word_tokenize(text)
    freq = Counter(word for word in words if word not in STOPWORDS and len(word) > 2)
    max_freq = max(freq.values(), default=1)
    for word in freq:
        freq[word] /= max_freq

    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence)
        score = sum(freq.get(word, 0) for word in sentence_words)
        length_penalty = math.log(len(sentence_words) + 1, 10)
        position_bonus = 1 / (idx + 1)
        sentence_scores[sentence] = score / (length_penalty or 1) + position_bonus

    ranked_sentences = sorted(
        sentence_scores, key=sentence_scores.get, reverse=True
    )[:max_sentences]
    ranked_sentences.sort(key=sentences.index)
    summary = " ".join(ranked_sentences)
    compression_ratio = len(summary) / max(len(text), 1)
    return summary, ranked_sentences, round(compression_ratio, 4)


def sentiment(text: str):
    tokens = word_tokenize(text)
    counts = Counter(tokens)
    pos_score = sum(counts[word] for word in POSITIVE_TERMS)
    neg_score = sum(counts[word] for word in NEGATIVE_TERMS)
    total = pos_score + neg_score + sum(counts.values()) * 0.25

    positive_ratio = pos_score / total if total else 0
    negative_ratio = neg_score / total if total else 0
    neutral_ratio = 1 - (positive_ratio + negative_ratio)
    polarity = positive_ratio - negative_ratio
    label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"

    top_pos = [word for word, _ in counts.most_common() if word in POSITIVE_TERMS][:5]
    top_neg = [word for word, _ in counts.most_common() if word in NEGATIVE_TERMS][:5]

    return {
        "label": label,
        "polarity": round(polarity, 4),
        "positive_score": round(positive_ratio, 4),
        "negative_score": round(negative_ratio, 4),
        "neutral_score": round(neutral_ratio, 4),
        "top_positive_terms": top_pos,
        "top_negative_terms": top_neg,
    }


def keywords(text: str, top_k: int) -> Tuple[List[str], List[float]]:
    sentences = sentence_tokenize(text)
    words = [word for word in word_tokenize(text) if word not in STOPWORDS]
    freq = Counter(words)
    degree = defaultdict(int)

    for sentence in sentences:
        unique_terms = set(
            word for word in word_tokenize(sentence) if word not in STOPWORDS
        )
        for term in unique_terms:
            degree[term] += len(unique_terms)

    scores = {term: degree[term] / freq[term] for term in freq}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    keywords_list, keyword_scores = zip(*ranked) if ranked else ([], [])
    return list(keywords_list), [round(score, 4) for score in keyword_scores]


def tfidf(documents: List[str], top_k: int, ngram: int) -> Tuple[List[str], List[List[str]]]:
    tokenized_docs = [
        [word for word in word_tokenize(doc) if word not in STOPWORDS] for doc in documents
    ]
    vocab_counter = Counter()
    doc_freq = Counter()
    ngram_docs = []

    for tokens in tokenized_docs:
        ngrams = []
        for n in range(1, ngram + 1):
            for idx in range(len(tokens) - n + 1):
                term = " ".join(tokens[idx : idx + n])
                ngrams.append(term)
                vocab_counter[term] += 1
        ngram_docs.append(ngrams)
        doc_freq.update(set(ngrams))

    total_docs = len(documents)
    tfidf_scores = []
    vocabulary = sorted(vocab_counter.keys())

    for ngrams in ngram_docs:
        counts = Counter(ngrams)
        doc_scores = {}
        for term, count in counts.items():
            tf = count / len(ngrams)
            idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1
            doc_scores[term] = tf * idf
        tfidf_scores.append(doc_scores)

    document_top_terms = []
    for doc_scores in tfidf_scores:
        top_terms = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        document_top_terms.append([term for term, _ in top_terms])

    return vocabulary, document_top_terms

