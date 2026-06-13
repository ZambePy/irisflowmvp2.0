# filter.py — IrisFlow 2.0
"""
One Euro Filter — suavização adaptativa do gaze em tempo real.

Parâmetros recomendados para gaze com webcam:
    min_cutoff = 1.0
    beta       = 0.007
"""

from __future__ import annotations
import math


class _LP:
    def __init__(self) -> None:
        self._y: float | None = None

    def filter(self, x: float, a: float) -> float:
        self._y = x if self._y is None else a * x + (1 - a) * self._y
        return self._y

    def last(self) -> float | None:
        return self._y


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self._mc = min_cutoff
        self._b  = beta
        self._dc = d_cutoff
        self._x  = _LP()
        self._dx = _LP()
        self._t: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * dt))

    def filter(self, x: float, t: float) -> float:
        dt   = max((t - self._t) if self._t else 1/30, 1e-6)
        self._t = t
        prev = self._x.last()
        dx   = 0.0 if prev is None else (x - prev) / dt
        edx  = self._dx.filter(dx, self._alpha(self._dc, dt))
        return self._x.filter(x, self._alpha(self._mc + self._b * abs(edx), dt))

    def reset(self) -> None:
        self._x = _LP(); self._dx = _LP(); self._t = None


class GazeFilter:
    """One Euro Filter aplicado independentemente em X e Y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)

    def filter(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self._fx.filter(x, t), self._fy.filter(y, t)

    def reset(self) -> None:
        self._fx.reset(); self._fy.reset()