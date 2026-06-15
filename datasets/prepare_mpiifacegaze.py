# datasets/prepare_mpiifacegaze.py — IrisFlow 2.0
"""
Preprocessa MPIIFaceGaze: crops MediaPipe + CSV unificado.

Uso:
    python datasets/prepare_mpiifacegaze.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import scipy.io
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tracker.head_pose import HeadPoseEstimator

_mp_face = mp.solutions.face_mesh

_LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_IDX = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

_PARTICIPANTS = [f"p{i:02d}" for i in range(15)]  # p00 a p14

_CSV_COLUMNS = [
    "participant", "img_path", "crop_dir", "stem",
    "gaze_pitch", "gaze_yaw",
    "head_yaw", "head_pitch", "head_roll",
    "face_bbox_x1", "face_bbox_y1", "face_bbox_x2", "face_bbox_y2",
    "frame_w", "frame_h",
]

_HALF_PI = math.pi / 2.0
_PI      = math.pi


def _bbox_from_landmarks(landmarks, indices: list[int], w: int, h: int, pad: float):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    x1 = max(0, int((min(xs) - pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w, int((max(xs) + pad) * w))
    y2 = min(h, int((max(ys) + pad) * h))
    return x1, y1, x2, y2


def _load_camera_matrix(p_dir: Path) -> np.ndarray | None:
    mat_path = p_dir / "Calibration" / "Camera.mat"
    if not mat_path.exists():
        return None
    mat = scipy.io.loadmat(str(mat_path))
    return mat.get("cameraMatrix", None)


def process_participant(pid: str, data_root: Path) -> tuple[list[dict], int, int]:
    """Processa um participante. Retorna (csv_rows, total_frames, detected_frames)."""
    p_dir    = data_root / pid
    txt_path = p_dir / f"{pid}.txt"
    if not txt_path.exists():
        return [], 0, 0

    crops_dir = data_root.parent / "crops" / pid
    crops_dir.mkdir(parents=True, exist_ok=True)

    cam_K = _load_camera_matrix(p_dir)

    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    face_mesh = _mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    metadata = {}
    csv_rows = []
    total    = len(lines)
    detected = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 11:
            continue

        rel_path   = parts[0]           # e.g. day01/0005.jpg
        gaze_pitch = float(parts[9])
        gaze_yaw   = float(parts[10])

        img_path = p_dir / rel_path
        if not img_path.exists():
            continue

        rel_p = Path(rel_path)
        day   = rel_p.parent.name        # day01
        stem  = f"{day}_{rel_p.stem}"    # day01_0005

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            continue

        detected += 1
        lm = results.multi_face_landmarks[0].landmark

        face_bbox = _bbox_from_landmarks(lm, list(range(468)), w_img, h_img, pad=0.1)
        l_bbox    = _bbox_from_landmarks(lm, _LEFT_EYE_IDX,   w_img, h_img, pad=0.2)
        r_bbox    = _bbox_from_landmarks(lm, _RIGHT_EYE_IDX,  w_img, h_img, pad=0.2)

        fx1, fy1, fx2, fy2 = face_bbox
        lx1, ly1, lx2, ly2 = l_bbox
        rx1, ry1, rx2, ry2 = r_bbox

        face_crop  = img_bgr[fy1:fy2, fx1:fx2]
        left_crop  = img_bgr[ly1:ly2, lx1:lx2]
        right_crop = img_bgr[ry1:ry2, rx1:rx2]

        if any(c.size == 0 for c in [face_crop, left_crop, right_crop]):
            detected -= 1
            continue

        # Head pose com câmera real se disponível
        estimator = HeadPoseEstimator(w_img, h_img)
        if cam_K is not None:
            estimator._K = np.array([
                [float(cam_K[0, 0]), 0.0,               float(cam_K[0, 2])],
                [0.0,                float(cam_K[1, 1]), float(cam_K[1, 2])],
                [0.0,                0.0,                1.0               ],
            ], dtype=np.float64)

        pose = estimator.estimate(lm)
        yaw, pitch, roll = pose if pose is not None else (0.0, 0.0, 0.0)

        cv2.imwrite(str(crops_dir / f"{stem}_face.jpg"),  face_crop)
        cv2.imwrite(str(crops_dir / f"{stem}_left.jpg"),  left_crop)
        cv2.imwrite(str(crops_dir / f"{stem}_right.jpg"), right_crop)

        metadata[stem] = {
            "yaw":      yaw,
            "pitch":    pitch,
            "roll":     roll,
            "face_bbox": list(face_bbox),
            "frame_w":  w_img,
            "frame_h":  h_img,
        }

        csv_rows.append({
            "participant":  pid,
            "img_path":     str(img_path.resolve()),
            "crop_dir":     str(crops_dir.resolve()),
            "stem":         stem,
            "gaze_pitch":   gaze_pitch / _HALF_PI,
            "gaze_yaw":     gaze_yaw   / _HALF_PI,
            "head_yaw":     yaw        / _PI,
            "head_pitch":   pitch      / _PI,
            "head_roll":    roll       / _PI,
            "face_bbox_x1": fx1,
            "face_bbox_y1": fy1,
            "face_bbox_x2": fx2,
            "face_bbox_y2": fy2,
            "frame_w":      w_img,
            "frame_h":      h_img,
        })

    face_mesh.close()

    with open(crops_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return csv_rows, total, detected


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessa MPIIFaceGaze")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(__file__).resolve().parent / "MPIIFaceGaze" / "Data"
    if not data_root.exists():
        print(f"[ERRO] Dataset não encontrado: {data_root}", file=sys.stderr)
        sys.exit(1)

    participants = [pid for pid in _PARTICIPANTS if (data_root / pid).exists()]

    all_rows       = []
    total_frames   = 0
    total_detected = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_participant, pid, data_root): pid
            for pid in participants
        }
        with tqdm(total=len(futures), desc="Participantes") as pbar:
            for future in as_completed(futures):
                rows, total, detected = future.result()
                all_rows.extend(rows)
                total_frames   += total
                total_detected += detected
                pbar.update(1)

    csv_path = data_root.parent / "mpiifacegaze.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    pct = 100.0 * total_detected / total_frames if total_frames else 0.0
    print(f"\nTotal de frames processados : {total_frames}")
    print(f"Frames com rosto detectado  : {total_detected} ({pct:.1f}%)")
    print(f"CSV gerado em               : {csv_path}")


if __name__ == "__main__":
    main()
