# calibration.py — IrisFlow 2.0
"""
Calibração de 9 pontos — converte gaze em cm para pixels na tela.

Fluxo:
    1. Usuário olha para 9 pontos conhecidos na tela
    2. Para cada ponto: coletar N predições do modelo (em cm)
    3. fit() → aprende a transformação afim cm → pixels
    4. convert() → aplica em tempo real

A transformação afim cobre translação, escala e rotação leve —
suficiente para compensar variações de distância e postura
entre sessões sem re-treinar o MLP.
"""

from __future__ import annotations

import numpy as np


class GazeCalibrator:
    """
    Calibração leve por regressão afim (cm → pixels).

    Não modifica o modelo — aprende apenas a transformação
    do espaço da câmera para o espaço da tela do usuário.

    Uso:
        cal = GazeCalibrator(screen_w=1920, screen_h=1080)

        # Para cada ponto de calibração:
        cal.add_point(
            gaze_cm=(x_cm, y_cm),   # predição média do modelo
            target_px=(tx, ty),     # posição conhecida na tela
        )

        cal.fit()                   # requer >= 4 pontos

        # Em tempo real:
        x_px, y_px = cal.convert(gaze_cm_x, gaze_cm_y)
    """

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._gaze_cm:   list[tuple[float, float]] = []
        self._target_px: list[tuple[float, float]] = []
        self._A: np.ndarray | None = None  # matriz afim (2, 3)

    def add_point(
        self,
        gaze_cm:   tuple[float, float],
        target_px: tuple[float, float],
    ) -> None:
        if len(self._gaze_cm) >= 9:
            raise ValueError("Máximo de 9 pontos de calibração.")
        self._gaze_cm.append(gaze_cm)
        self._target_px.append(target_px)

    def point_count(self) -> int:
        return len(self._gaze_cm)

    def fit(self) -> float:
        """
        Ajusta transformação afim por mínimos quadrados.

        Returns:
            MAE em pixels no conjunto de calibração.

        Raises:
            ValueError: se houver menos de 4 pontos.
        """
        n = len(self._gaze_cm)
        if n < 4:
            raise ValueError(f"Mínimo de 4 pontos — {n} fornecidos.")

        # Fonte: pontos em cm (com coluna de bias)
        src = np.array([[x, y, 1.0] for x, y in self._gaze_cm], dtype=np.float64)
        # Destino: pixels na tela
        dst = np.array(self._target_px, dtype=np.float64)

        # Resolver: dst = src @ A.T  →  A = (src.T @ src)^-1 @ src.T @ dst
        self._A, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)

        # MAE de calibração
        pred = src @ self._A
        mae  = float(np.mean(np.sqrt(((pred - dst)**2).sum(axis=1))))
        return mae

    def convert(self, gaze_x_cm: float, gaze_y_cm: float) -> tuple[float, float]:
        """
        Converte predição do modelo (cm) → pixels na tela.

        Raises:
            RuntimeError: se fit() não foi chamado ainda.
        """
        if self._A is None:
            raise RuntimeError("Chame fit() antes de convert().")

        src  = np.array([gaze_x_cm, gaze_y_cm, 1.0], dtype=np.float64)
        pred = src @ self._A  # (2,)

        x = float(np.clip(pred[0], 0, self._screen_w))
        y = float(np.clip(pred[1], 0, self._screen_h))
        return x, y

    def reset(self) -> None:
        self._gaze_cm.clear()
        self._target_px.clear()
        self._A = None