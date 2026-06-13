# model.py — IrisFlow 2.0
"""
IrisGazeNet v3

Arquitetura:
    left_eye  (3, 112, 112) → MobileNetV3-Small → (576,)
    right_eye (3, 112, 112) → MobileNetV3-Small → (576,)
    face      (3, 224, 224) → MobileNetV3-Small → (576,)
    face_grid (1, 25, 25)   → Flatten           → (625,)
    head_pose               → [yaw, pitch, roll] → (3,)

    concat → (576×3 + 625 + 3) = 2356
    MLP    → 1024 → 512 → 256 → 2
    output → (gaze_x_cm, gaze_y_cm) relativo à câmera

Os 3 backbones começam com pesos ImageNet idênticos mas são
instâncias independentes — cada canal fine-tuna seus próprios pesos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def _make_mobilenetv3() -> tuple[nn.Module, int]:
    """
    Retorna o backbone MobileNetV3-Small sem o classifier.
    Output: (B, 576) após avgpool.
    """
    backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    features = backbone.features
    avgpool  = backbone.avgpool
    dim      = 576
    return nn.Sequential(features, avgpool, nn.Flatten(1)), dim


class IrisGazeNet(nn.Module):
    """
    Modelo completo de gaze estimation.

    Entrada:
        left_eye:  (B, 3, 112, 112)
        right_eye: (B, 3, 112, 112)
        face:      (B, 3, 224, 224)
        face_grid: (B, 1, 25, 25)
        head_pose: (B, 3)  — [yaw, pitch, roll] em radianos

    Saída:
        (B, 2) — [gaze_x_cm, gaze_y_cm] relativo à câmera
    """

    EMBED_DIM    = 576
    GRID_DIM     = 625   # 25×25
    POSE_DIM     = 3
    INPUT_DIM    = EMBED_DIM * 3 + GRID_DIM + POSE_DIM  # 2356

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        # 3 backbones independentes — mesmo ponto de partida, pesos separados
        self.left_eye_net,  _ = _make_mobilenetv3()
        self.right_eye_net, _ = _make_mobilenetv3()
        self.face_net,      _ = _make_mobilenetv3()

        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2),   # (gaze_x_cm, gaze_y_cm)
        )

    def forward(
        self,
        left_eye:  torch.Tensor,   # (B, 3, 112, 112)
        right_eye: torch.Tensor,   # (B, 3, 112, 112)
        face:      torch.Tensor,   # (B, 3, 224, 224)
        face_grid: torch.Tensor,   # (B, 1, 25, 25)
        head_pose: torch.Tensor,   # (B, 3)
    ) -> torch.Tensor:

        e_left  = self.left_eye_net(left_eye)          # (B, 576)
        e_right = self.right_eye_net(right_eye)        # (B, 576)
        e_face  = self.face_net(face)                  # (B, 576)
        grid    = face_grid.flatten(1)                 # (B, 625)

        x = torch.cat([e_left, e_right, e_face, grid, head_pose], dim=1)  # (B, 2356)
        return self.mlp(x)                             # (B, 2)

    @torch.no_grad()
    def predict(
        self,
        left_eye:  torch.Tensor,
        right_eye: torch.Tensor,
        face:      torch.Tensor,
        face_grid: torch.Tensor,
        head_pose: torch.Tensor,
    ) -> tuple[float, float]:
        """Single-sample — retorna (gaze_x_cm, gaze_y_cm)."""
        self.eval()
        out = self.forward(left_eye, right_eye, face, face_grid, head_pose)
        return float(out[0, 0]), float(out[0, 1])

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.state_dict(),
            "architecture": "IrisGazeNet_v3",
            "input_dim": self.INPUT_DIM,
        }, path)

    @classmethod
    def load(cls, path: str) -> "IrisGazeNet":
        data  = torch.load(path, map_location="cpu")
        model = cls()
        model.load_state_dict(data["model_state"])
        return model