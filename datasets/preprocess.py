# preprocess.py — IrisFlow 2.0
# preprocess.py — IrisFlow 2.0

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

from tracker.head_pose import HeadPoseEstimator

_mp_face = mp.solutions.face_mesh

_LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249,
                   263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_IDX = [33,  7,   163, 144, 145, 153, 154, 155,
                   133, 173, 157, 158, 159, 160, 161, 246]


def _bbox(landmarks, indices, w, h, pad=0.2):
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    x1 = max(0, int((min(xs) - pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w, int((max(xs) + pad) * w))
    y2 = min(h, int((max(ys) + pad) * h))
    return x1, y1, x2, y2


def _process_frame(
    img_path: Path,
    out_dir: Path,
    face_mesh,
    pose_estimator: HeadPoseEstimator,
) -> dict | None:
    """
    Processa um único frame:
        - Detecta rosto e olhos com MediaPipe
        - Salva crops: {stem}_left.jpg, {stem}_right.jpg, {stem}_face.jpg
        - Retorna metadata: yaw, pitch, roll, face_bbox, frame_w, frame_h

    Retorna None se MediaPipe não detectar rosto.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res  = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None

    lm   = res.multi_face_landmarks[0].landmark
    stem = img_path.stem

    # Face bbox
    xs = [l.x for l in lm]
    ys = [l.y for l in lm]
    fx1 = max(0, int((min(xs) - 0.1) * w))
    fy1 = max(0, int((min(ys) - 0.1) * h))
    fx2 = min(w, int((max(xs) + 0.1) * w))
    fy2 = min(h, int((max(ys) + 0.1) * h))

    # Eye bboxes
    lx1, ly1, lx2, ly2 = _bbox(lm, _LEFT_EYE_IDX,  w, h)
    rx1, ry1, rx2, ry2 = _bbox(lm, _RIGHT_EYE_IDX, w, h)

    crops = {
        "face":  img[fy1:fy2, fx1:fx2],
        "left":  img[ly1:ly2, lx1:lx2],
        "right": img[ry1:ry2, rx1:rx2],
    }

    for name, crop in crops.items():
        if crop.size == 0:
            return None
        cv2.imwrite(str(out_dir / f"{stem}_{name}.jpg"), crop)

    # Head pose via solvePnP
    pose = pose_estimator.estimate(lm)
    yaw, pitch, roll = pose if pose else (0.0, 0.0, 0.0)

    return {
        "yaw":       yaw,
        "pitch":     pitch,
        "roll":      roll,
        "face_bbox": [fx1, fy1, fx2, fy2],
        "frame_w":   w,
        "frame_h":   h,
    }


def process_session(session_dir: Path, crops_root: Path) -> tuple[int, int]:
    """
    Processa todos os frames de uma sessão do GazeCapture.

    Salva crops em crops/{session_id}/
    Salva metadata em crops/{session_id}/metadata.json
    """
    frames_dir = session_dir / "frames"
    if not frames_dir.exists():
        return 0, 0

    out_dir = crops_root / session_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ler resolução do frame a partir do primeiro frame disponível
    first = next(frames_dir.glob("*.jpg"), None)
    frame_w, frame_h = 640, 480
    if first:
        img = cv2.imread(str(first))
        if img is not None:
            frame_h, frame_w = img.shape[:2]

    face_mesh = _mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )
    pose_estimator = HeadPoseEstimator(frame_w=frame_w, frame_h=frame_h)

    imgs     = sorted(frames_dir.glob("*.jpg"))
    metadata = {}
    ok       = 0

    for img_path in imgs:
        result = _process_frame(img_path, out_dir, face_mesh, pose_estimator)
        if result:
            metadata[img_path.stem] = result
            ok += 1

    # Salvar metadata da sessão
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return ok, len(imgs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IrisFlow 2.0 — pré-processamento offline do GazeCapture"
    )
    parser.add_argument("--root",    required=True, help="Raiz do GazeCapture")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root      = Path(args.root)
    crops_dir = root / "crops"
    crops_dir.mkdir(exist_ok=True)

    sessions = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name != "crops"
    ])

    print(f"Processando {len(sessions)} sessões com {args.workers} workers...")
    print("Isso pode levar alguns minutos dependendo do tamanho do dataset.\n")

    total_ok = total_frames = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_session, s, crops_dir): s
            for s in sessions
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sessões"):
            ok, n = fut.result()
            total_ok     += ok
            total_frames += n

    pct = total_ok / total_frames * 100 if total_frames else 0
    print(f"\nConcluído: {total_ok}/{total_frames} frames ({pct:.1f}%) detectados pelo MediaPipe")
    print(f"Crops salvos em: {crops_dir}")
    print("\nPróximo passo: python core/train.py --gazecapture datasets/GazeCapture")


if __name__ == "__main__":
    main()