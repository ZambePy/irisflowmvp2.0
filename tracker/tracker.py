# tracker.py — IrisFlow 2.0
"""
GazeTracker — pipeline completo de captura em tempo real.

Fluxo por frame:
    Webcam → MediaPipe FaceMesh → Face Crop → Eye Crops
           → Face Grid (25×25) → Head Pose
           → tensors prontos para IrisGazeNet.predict()
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

from head_pose import HeadPoseEstimator

_mp_face = mp.solutions.face_mesh

# Índices MediaPipe para recortes
_LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_IDX = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_FACE_OVAL_IDX = list(range(0, 468))  # todos os landmarks para bbox do rosto

_EYE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_FACE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _bbox_from_landmarks(landmarks: list, indices: list, w: int, h: int, pad: float = 0.15):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    x1 = max(0, int((min(xs) - pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w, int((max(xs) + pad) * w))
    y2 = min(h, int((max(ys) + pad) * h))
    return x1, y1, x2, y2


def _make_face_grid(
    face_bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    grid_size: int = 25,
) -> torch.Tensor:
    """
    Gera um mapa binário 25×25 indicando a posição do rosto no frame.

    O face grid é usado pelo modelo para inferir a posição espacial
    do rosto na imagem — informação que os crops isolados perdem.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Mapear bbox do rosto para o grid
    gx1 = int(fx1 / frame_w * grid_size)
    gy1 = int(fy1 / frame_h * grid_size)
    gx2 = int(fx2 / frame_w * grid_size)
    gy2 = int(fy2 / frame_h * grid_size)

    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(grid_size, gx2), min(grid_size, gy2)
    grid[gy1:gy2, gx1:gx2] = 1.0

    return torch.from_numpy(grid).unsqueeze(0)  # (1, 25, 25)


class FrameData:
    """Resultado do processamento de um único frame."""

    __slots__ = [
        "left_eye", "right_eye", "face",
        "face_grid", "head_pose",
        "landmarks", "face_bbox",
    ]

    def __init__(
        self,
        left_eye:  torch.Tensor,   # (1, 3, 112, 112)
        right_eye: torch.Tensor,   # (1, 3, 112, 112)
        face:      torch.Tensor,   # (1, 3, 224, 224)
        face_grid: torch.Tensor,   # (1, 1, 25, 25)
        head_pose: torch.Tensor,   # (1, 3)
        landmarks: list,
        face_bbox: tuple[int, int, int, int],
    ) -> None:
        self.left_eye  = left_eye
        self.right_eye = right_eye
        self.face      = face
        self.face_grid = face_grid
        self.head_pose = head_pose
        self.landmarks = landmarks
        self.face_bbox = face_bbox


class GazeTracker:
    """
    Pipeline de captura em tempo real.

    Uso:
        tracker = GazeTracker(camera_index=0)
        tracker.start()
        while True:
            data = tracker.process_frame()
            if data is not None:
                gaze_x, gaze_y = model.predict(
                    data.left_eye, data.right_eye,
                    data.face, data.face_grid, data.head_pose,
                )
        tracker.stop()
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
        self._w = frame_w
        self._h = frame_h

        self._face_mesh = _mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose_estimator = HeadPoseEstimator(frame_w, frame_h)

    def process_frame(self) -> FrameData | None:
        ret, frame_bgr = self._cap.read()
        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results   = self._face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # Face crop
        face_bbox = _bbox_from_landmarks(lm, list(range(468)), self._w, self._h, pad=0.1)
        fx1, fy1, fx2, fy2 = face_bbox
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        face_crop = frame_bgr[fy1:fy2, fx1:fx2]

        # Eye crops
        lx1, ly1, lx2, ly2 = _bbox_from_landmarks(lm, _LEFT_EYE_IDX,  self._w, self._h, pad=0.2)
        rx1, ry1, rx2, ry2 = _bbox_from_landmarks(lm, _RIGHT_EYE_IDX, self._w, self._h, pad=0.2)

        left_crop  = frame_bgr[ly1:ly2, lx1:lx2]
        right_crop = frame_bgr[ry1:ry2, rx1:rx2]

        if any(c.size == 0 for c in [face_crop, left_crop, right_crop]):
            return None

        # Face grid
        face_grid = _make_face_grid(face_bbox, self._w, self._h)

        # Head pose
        pose = self._pose_estimator.estimate(lm)
        if pose is None:
            pose = (0.0, 0.0, 0.0)
        yaw, pitch, roll = pose

        # Tensors
        left_t  = _EYE_TRANSFORM(cv2.cvtColor(left_crop,  cv2.COLOR_BGR2RGB)).unsqueeze(0)
        right_t = _EYE_TRANSFORM(cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        face_t  = _FACE_TRANSFORM(cv2.cvtColor(face_crop,  cv2.COLOR_BGR2RGB)).unsqueeze(0)
        pose_t  = torch.tensor([[yaw, pitch, roll]], dtype=torch.float32)

        return FrameData(
            left_eye=left_t,
            right_eye=right_t,
            face=face_t,
            face_grid=face_grid.unsqueeze(0),
            head_pose=pose_t,
            landmarks=lm,
            face_bbox=face_bbox,
        )

    def stop(self) -> None:
        self._cap.release()