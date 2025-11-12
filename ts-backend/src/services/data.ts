interface NormalizeStats {
  mean: number;
  variance: number;
  stdDev: number;
  min: number;
  max: number;
  median: number;
  q1: number;
  q3: number;
  iqr: number;
}

export interface NormalizeResult {
  zScores: number[];
  minMax: number[];
  robust: number[];
  stats: NormalizeStats;
}

export interface RegressionResult {
  coefficients: number[];
  intercept: number;
  rSquared: number;
  predictions: number[];
}

export interface CorrelationResult {
  pearson: number;
  spearman: number;
  kendall: number;
}

export interface FeatureSummary {
  mean: number;
  min: number;
  max: number;
  stdDev: number;
}

export function normalize(values: number[]): NormalizeResult {
  if (values.length === 0) {
    throw new Error("values array cannot be empty");
  }

  const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
  const variance =
    values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const ordered = [...values].sort((a, b) => a - b);
  const median = percentile(ordered, 50);
  const q1 = percentile(ordered, 25);
  const q3 = percentile(ordered, 75);
  const iqr = q3 - q1;
  const robustScale = Math.abs(iqr) > 1e-9 ? iqr : 1;

  const zScores = values.map((value) => ((value - mean) / std) || 0);
  const minMax = values.map((value) => (value - min) / range);
  const robust = values.map((value) => (value - median) / robustScale);

  return {
    zScores: zScores.map((value) => Number(value.toFixed(4))),
    minMax: minMax.map((value) => Number(value.toFixed(4))),
    robust: robust.map((value) => Number(value.toFixed(4))),
    stats: {
      mean: Number(mean.toFixed(4)),
      variance: Number(variance.toFixed(4)),
      stdDev: Number(std.toFixed(4)),
      min,
      max,
      median: Number(median.toFixed(4)),
      q1: Number(q1.toFixed(4)),
      q3: Number(q3.toFixed(4)),
      iqr: Number(iqr.toFixed(4)),
    },
  };
}

export function linearRegression(
  features: number[][],
  targets: number[],
): RegressionResult {
  if (features.length !== targets.length) {
    throw new Error("features and targets length mismatch");
  }
  if (features.length === 0) {
    throw new Error("at least one observation required");
  }

  const m = features.length;
  const n = features[0]?.length ?? 0;
  if (n === 0) {
    throw new Error("feature vectors cannot be empty");
  }
  const augmented = features.map((row) => [...row, 1]);

  const xtx: number[][] = Array.from({ length: n + 1 }, () =>
    new Array(n + 1).fill(0),
  );
  const xty: number[] = new Array(n + 1).fill(0);

  for (let idx = 0; idx < augmented.length; idx += 1) {
    const row = augmented[idx];
    const y = targets[idx];
    if (!row || typeof y === "undefined") {
      throw new Error("invalid observation encountered");
    }
    for (let i = 0; i <= n; i += 1) {
      const value = row[i];
      if (value === undefined) continue;
      const xtxRow = xtx[i];
      if (!xtxRow || i >= xty.length) continue;
      xty[i] = (xty[i] ?? 0) + value * y;
      for (let j = 0; j <= n; j += 1) {
        const valueJ = row[j];
        if (valueJ === undefined || j >= xtxRow.length) continue;
        xtxRow[j] = (xtxRow[j] ?? 0) + value * valueJ;
      }
    }
  }

  const solved = gaussianElimination(xtx, xty);
  if (typeof solved[n] === "undefined") {
    throw new Error("regression solver failed");
  }
  const intercept = solved[n];
  const weights = solved.slice(0, n);

  const predictions = features.map((row) =>
    row.reduce((acc, value, idx) => acc + value * (weights[idx] ?? 0), intercept),
  );

  const meanY = targets.reduce((acc, value) => acc + value, 0) / m;
  const ssTot = targets.reduce((acc, value) => acc + (value - meanY) ** 2, 0);
  const ssRes = targets.reduce((acc, value, idx) => {
    const prediction = predictions[idx] ?? 0;
    return acc + (value - prediction) ** 2;
  }, 0);
  const rSquared = ssTot ? 1 - ssRes / ssTot : 0;

  return {
    coefficients: weights.map((value) => Number(value.toFixed(6))),
    intercept: Number(intercept.toFixed(6)),
    rSquared: Number(rSquared.toFixed(6)),
    predictions: predictions.map((value) => Number(value.toFixed(6))),
  };
}

export function correlations(
  seriesA: number[],
  seriesB: number[],
): CorrelationResult {
  if (seriesA.length !== seriesB.length) {
    throw new Error("Series must have equal length");
  }
  if (seriesA.length < 2) {
    throw new Error("Series must contain at least two values");
  }

  const pearson = computePearson(seriesA, seriesB);
  const rankA = rank(seriesA);
  const rankB = rank(seriesB);
  const spearman = computePearson(rankA, rankB);
  const kendall = computeKendall(seriesA, seriesB);

  return {
    pearson: Number(pearson.toFixed(4)),
    spearman: Number(spearman.toFixed(4)),
    kendall: Number(kendall.toFixed(4)),
  };
}

function percentile(values: number[], percentileRank: number): number {
  if (values.length === 0) return 0;
  const position = (values.length - 1) * (percentileRank / 100);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return values[lower];
  const weight = position - lower;
  return values[lower] * (1 - weight) + values[upper] * weight;
}

function gaussianElimination(matrix: number[][], vector: number[]): number[] {
  const size = vector.length;
  const mat = matrix.map((row) => [...row]);
  const vec = [...vector];

  for (let i = 0; i < size; i += 1) {
    let pivotIndex = i;
    for (let j = i + 1; j < size; j += 1) {
      if (Math.abs(mat[j]?.[i] ?? 0) > Math.abs(mat[pivotIndex]?.[i] ?? 0)) {
        pivotIndex = j;
      }
    }

    const pivotRow = mat[pivotIndex];
    const currentRow = mat[i];
    if (!pivotRow || !currentRow) throw new Error("Singular matrix detected");
    if (pivotIndex !== i) {
      mat[i] = pivotRow;
      mat[pivotIndex] = currentRow;
      const tmpVec = vec[i];
      vec[i] = vec[pivotIndex] ?? 0;
      vec[pivotIndex] = tmpVec ?? 0;
    }

    const rowI = mat[i];
    if (!rowI) throw new Error("Row access failed");
    let pivot = rowI[i] ?? 0;
    if (Math.abs(pivot) < 1e-9) {
      pivot = pivot >= 0 ? 1e-9 : -1e-9;
    }

    for (let j = i; j < size; j += 1) {
      const current = typeof rowI[j] === "number" ? rowI[j] : 0;
      rowI[j] = current / pivot;
    }
    const vecCurrent = typeof vec[i] === "number" ? vec[i] : 0;
    vec[i] = vecCurrent / pivot;

    for (let k = 0; k < size; k += 1) {
      if (k === i) continue;
      const targetRow = mat[k];
      if (!targetRow) throw new Error("Row access failed");
      const multiplier = targetRow[i] ?? 0;
      for (let j = i; j < size; j += 1) {
        const targetValue = typeof targetRow[j] === "number" ? targetRow[j] : 0;
        const rowValue = typeof rowI[j] === "number" ? rowI[j] : 0;
        targetRow[j] = targetValue - multiplier * rowValue;
      }
      const vecValue = typeof vec[k] === "number" ? vec[k] : 0;
      const currentPivotVec = typeof vec[i] === "number" ? vec[i] : 0;
      vec[k] = vecValue - multiplier * currentPivotVec;
    }
  }

  return vec;
}

function computePearson(seriesA: number[], seriesB: number[]): number {
  const meanA =
    seriesA.reduce((acc, value) => acc + value, 0) / seriesA.length;
  const meanB =
    seriesB.reduce((acc, value) => acc + value, 0) / seriesB.length;
  let numerator = 0;
  let denomA = 0;
  let denomB = 0;

  for (let i = 0; i < seriesA.length; i += 1) {
    const diffA = seriesA[i]! - meanA;
    const diffB = seriesB[i]! - meanB;
    numerator += diffA * diffB;
    denomA += diffA ** 2;
    denomB += diffB ** 2;
  }
  const denominator = Math.sqrt(denomA * denomB) || 1;
  return numerator / denominator;
}

function rank(values: number[]): number[] {
  const pairs = values.map((value, index) => ({ value, index }));
  pairs.sort((a, b) => a.value - b.value);

  const ranks = Array(values.length).fill(0);
  let i = 0;
  while (i < pairs.length) {
    let j = i;
    let rankSum = 0;
    while (j < pairs.length && pairs[j].value === pairs[i].value) {
      rankSum += j + 1;
      j += 1;
    }
    const avgRank = rankSum / (j - i);
    for (let k = i; k < j; k += 1) {
      ranks[pairs[k].index] = avgRank;
    }
    i = j;
  }
  return ranks;
}

function computeKendall(seriesA: number[], seriesB: number[]): number {
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < seriesA.length; i += 1) {
    for (let j = i + 1; j < seriesA.length; j += 1) {
      const signA = Math.sign(seriesA[i]! - seriesA[j]!);
      const signB = Math.sign(seriesB[i]! - seriesB[j]!);
      if (signA === signB) concordant += 1;
      else discordant += 1;
    }
  }
  const total = concordant + discordant;
  return total ? (concordant - discordant) / total : 0;
}

export function featureSummary(features: number[][]): FeatureSummary[] {
  if (features.length === 0) {
    throw new Error("feature matrix cannot be empty");
  }
  const cols = features[0]?.length ?? 0;
  if (cols === 0) {
    throw new Error("feature vectors must contain at least one value");
  }

  const columnData: number[][] = Array.from({ length: cols }, () => []);
  features.forEach((row, rowIdx) => {
    if (row.length !== cols) {
      throw new Error(`row ${rowIdx} has inconsistent length`);
    }
    row.forEach((value, colIdx) => {
      columnData[colIdx]?.push(value);
    });
  });

  return columnData.map((values) => {
    const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
    const variance =
      values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...values);
    const max = Math.max(...values);
    return {
      mean: Number(mean.toFixed(4)),
      min: Number(min.toFixed(4)),
      max: Number(max.toFixed(4)),
      stdDev: Number(stdDev.toFixed(4)),
    };
  });
}

