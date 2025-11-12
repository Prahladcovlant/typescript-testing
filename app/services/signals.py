from __future__ import annotations

import math
from typing import List


def fourier_transform(signal: List[float], sample_rate: float):
    n = len(signal)
    real_parts = []
    imag_parts = []
    magnitudes = []
    for k in range(n):
        real = sum(signal[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        imag = -sum(signal[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        real_parts.append(real)
        imag_parts.append(imag)
        magnitudes.append(math.sqrt(real ** 2 + imag ** 2))

    dominant_idx = max(range(n // 2), key=lambda k: magnitudes[k])
    dominant_frequency = dominant_idx * sample_rate / n
    energy = sum(mag ** 2 for mag in magnitudes) / n

    return {
        "magnitudes": [round(mag, 4) for mag in magnitudes[: n // 2]],
        "dominant_frequency": round(dominant_frequency, 4),
        "energy": round(energy, 4),
    }

