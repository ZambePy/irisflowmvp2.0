# head_pose.py — IrisFlow 2.0
"""
HeadPoseEstimator — yaw/pitch/roll via MediaPipe + solvePnP.

Usa 6 landmarks canônicos do rosto para estimar a pose 3D
da cabeça em relação à câmera.

Output: [yaw, pitch, roll] em radianos.
"""

from __future__ import annotations

import cv2
import numpy as np

# Índices dos 6 landmarks MediaPipe usados no solvePnP
# nariz, queixo, olho esq. ext., olho dir. ext., boca esq., boca dir.
_LM_IDX = [1, 152, 33, 263, 61, 291]

# Coordenadas 3D de referência (modelo genérico, mm)
_MODEL_3D = np.array([
    [  0.0,    0.0,   0.0],
    [  0.0,  -63.6, -12.5],
    [-43.3,   32.7, -26.0],
    [ 43.3,   32.7, -26.0],
    [-28.9,  -28.9, -24.1],
    [ 28.9,  -28.9, -24.1],
], dtype=np.float64)


class HeadPoseEstimator:

    def __init__(self, frame_w: int = 640, frame_h: int = 480) -> None:
        focal = float(frame_w)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        self._K = np.array([
            [focal,   0,  cx],
            [    0, focal, cy],
            [    0,     0,  1],
        ], dtype=np.float64)
        self._dist = np.zeros((4, 1), dtype=np.float64)
        self._w = frame_w
        self._h = frame_h

    def estimate(self, landmarks: list) -> tuple[float, float, float] | None:
        """
        Retorna (yaw, pitch, roll) em radianos ou None se solvePnP falhar.

        Args:
            landmarks: lista de NormalizedLandmark do MediaPipe FaceMesh
        """
        pts_2d = np.array([
            [landmarks[i].x * self._w, landmarks[i].y * self._h]
            for i in _LM_IDX
        ], dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _MODEL_3D, pts_2d, self._K, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        return self._euler(rmat)

    @staticmethod
    def _euler(R: np.ndarray) -> tuple[float, float, float]:
        sy = float(np.sqrt(R[0, 0]**2 + R[1, 0]**2))
        if sy > 1e-6:
            pitch = float(np.arctan2( R[2, 1], R[2, 2]))
            yaw   = float(np.arctan2(-R[2, 0], sy))
            roll  = float(np.arctan2( R[1, 0], R[0, 0]))
        else:
            pitch = float(np.arctan2(-R[1, 2], R[1, 1]))
            yaw   = float(np.arctan2(-R[2, 0], sy))
            roll  = 0.0
        return yaw, pitch, roll