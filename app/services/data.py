from __future__ import annotations

import math
from typing import List, Tuple


def normalize(values: List[float]) -> Tuple[List[float], List[float], dict]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    minimum = min(values)
    maximum = max(values)
    range_val = maximum - minimum or 1

    zscores = [(x - mean) / std if std else 0 for x in values]
    min_max_scaled = [(x - minimum) / range_val for x in values]
    statistics = {
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "std_dev": round(std, 4),
        "min": minimum,
        "max": maximum,
    }
    return zscores, min_max_scaled, statistics


def linear_regression(features: List[List[float]], targets: List[float]):
    if len(features) != len(targets):
        raise ValueError("features and targets length mismatch")

    m = len(features)
    n = len(features[0])
    augmented = [feat + [1.0] for feat in features]

    xtx = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    xty = [0.0 for _ in range(n + 1)]

    for row, y in zip(augmented, targets):
        for i in range(n + 1):
            xty[i] += row[i] * y
            for j in range(n + 1):
                xtx[i][j] += row[i] * row[j]

    coeffs = _gaussian_elimination(xtx, xty)
    intercept = coeffs[-1]
    weights = coeffs[:-1]
    predictions = [sum(w * x for w, x in zip(weights, feat)) + intercept for feat in features]

    mean_y = sum(targets) / m
    ss_tot = sum((y - mean_y) ** 2 for y in targets)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(targets, predictions))
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0

    return weights, intercept, r_squared, predictions


def correlations(series_a: List[float], series_b: List[float]):
    if len(series_a) != len(series_b):
        raise ValueError("Series must be same length")

    def mean(data):
        return sum(data) / len(data)

    mean_a = mean(series_a)
    mean_b = mean(series_b)
    diff_a = [x - mean_a for x in series_a]
    diff_b = [y - mean_b for y in series_b]
    covariance = sum(x * y for x, y in zip(diff_a, diff_b))
    std_a = math.sqrt(sum(x ** 2 for x in diff_a))
    std_b = math.sqrt(sum(y ** 2 for y in diff_b))
    pearson = covariance / (std_a * std_b) if std_a and std_b else 0

    rank_a = _rank(series_a)
    rank_b = _rank(series_b)
    spearman = correlations(rank_a, rank_b)["pearson"] if len(series_a) < 100 else pearson

    concordant = discordant = 0
    for i in range(len(series_a)):
        for j in range(i + 1, len(series_a)):
            sign_a = series_a[i] < series_a[j]
            sign_b = series_b[i] < series_b[j]
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    total_pairs = concordant + discordant
    kendall = (concordant - discordant) / total_pairs if total_pairs else 0

    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "kendall": round(kendall, 4),
    }


def _gaussian_elimination(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    for i in range(n):
        pivot = matrix[i][i]
        if abs(pivot) < 1e-9:
            for j in range(i + 1, n):
                if abs(matrix[j][i]) > abs(pivot):
                    matrix[i], matrix[j] = matrix[j], matrix[i]
                    vector[i], vector[j] = vector[j], vector[i]
                    pivot = matrix[i][i]
                    break
        factor = pivot or 1e-9
        for j in range(i, n):
            matrix[i][j] /= factor
        vector[i] /= factor

        for k in range(n):
            if k == i:
                continue
            multiplier = matrix[k][i]
            for j in range(i, n):
                matrix[k][j] -= multiplier * matrix[i][j]
            vector[k] -= multiplier * vector[i]

    return vector[:]


def _rank(values: List[float]) -> List[float]:
    sorted_pairs = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        total_rank = 0
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            total_rank += j + 1
            j += 1
        avg_rank = total_rank / (j - i)
        for k in range(i, j):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j
    return ranks

