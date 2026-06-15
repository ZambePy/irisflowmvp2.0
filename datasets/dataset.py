# dataset.py — IrisFlow 2.0
# dataset.py — IrisFlow 2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


# ── Transforms ────────────────────────────────────────────────────────────────

_EYE_T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_FACE_T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _make_face_grid(
    face_bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    grid_size: int = 25,
) -> torch.Tensor:
    """Gera mapa binário 25×25 indicando posição do rosto no frame."""
    fx1, fy1, fx2, fy2 = face_bbox
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    gx1 = int(fx1 / frame_w * grid_size)
    gy1 = int(fy1 / frame_h * grid_size)
    gx2 = int(fx2 / frame_w * grid_size)
    gy2 = int(fy2 / frame_h * grid_size)
    grid[
        max(0, gy1):min(grid_size, gy2),
        max(0, gx1):min(grid_size, gx2),
    ] = 1.0
    return torch.from_numpy(grid).unsqueeze(0)  # (1, 25, 25)


def _load_crops(crop_dir: Path, stem: str):
    """Carrega os 3 crops gerados pelo preprocess.py."""
    left  = cv2.imread(str(crop_dir / f"{stem}_left.jpg"))
    right = cv2.imread(str(crop_dir / f"{stem}_right.jpg"))
    face  = cv2.imread(str(crop_dir / f"{stem}_face.jpg"))
    return left, right, face


def _crops_to_tensors(left, right, face, face_bbox, frame_w, frame_h):
    """Converte crops numpy → tensors normalizados + face grid."""
    if left  is None: left  = np.zeros((112, 112, 3), dtype=np.uint8)
    if right is None: right = np.zeros((112, 112, 3), dtype=np.uint8)
    if face  is None: face  = np.zeros((224, 224, 3), dtype=np.uint8)

    return (
        _EYE_T( cv2.cvtColor(left,  cv2.COLOR_BGR2RGB)),
        _EYE_T( cv2.cvtColor(right, cv2.COLOR_BGR2RGB)),
        _FACE_T(cv2.cvtColor(face,  cv2.COLOR_BGR2RGB)),
        _make_face_grid(face_bbox, frame_w, frame_h),
    )


# ── GazeCapture ───────────────────────────────────────────────────────────────

class GazeCaptureDataset(Dataset):
    """
    Dataset GazeCapture (MIT) para treino do IrisGazeNet v3.

    Estrutura esperada após download e preprocess:
        datasets/GazeCapture/
            {session_id}/
                frames/
                    {frame_id:05d}.jpg   — frames brutos
                info.json                — metadados da sessão
            crops/
                {session_id}/
                    {frame_id:05d}_left.jpg
                    {frame_id:05d}_right.jpg
                    {frame_id:05d}_face.jpg

    Labels usados:
        XCam, YCam  — gaze em cm relativo à câmera (já fornecido pelo dataset)

    Os crops originais do GazeCapture NÃO são usados.
    Todos os crops são gerados offline pelo preprocess.py com MediaPipe,
    garantindo consistência com a pipeline de inferência em tempo real.

    Uso:
        ds = GazeCaptureDataset("datasets/GazeCapture", split="train")
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    """

    # Últimas N sessões reservadas para validação
    _VAL_SESSIONS = 50

    def __init__(
        self,
        root: str | Path,
        split: str = "train",   # "train" ou "val"
        augment: bool = False,
    ) -> None:
        self.root    = Path(root)
        self.augment = augment
        self.samples: list[dict] = []
        self._load(split)

    def _load(self, split: str) -> None:
        crops_dir = self.root / "crops"
        sessions  = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and d.name != "crops"
        ])

        # Split por sessão
        if split == "train":
            sessions = sessions[:-self._VAL_SESSIONS]
        else:
            sessions = sessions[-self._VAL_SESSIONS:]

        for session_dir in sessions:
            info_path = session_dir / "info.json"
            if not info_path.exists():
                continue

            with open(info_path) as f:
                info = json.load(f)

            # GazeCapture fornece XCam/YCam em cm — usar diretamente
            dot_info = info.get("dotInfo", {})
            x_cam_list = dot_info.get("XCam", [])
            y_cam_list = dot_info.get("YCam", [])
            is_val     = info.get("labelRecNum", [])

            frame_dir  = session_dir / "frames"
            session_crops = crops_dir / session_dir.name

            for i, is_valid in enumerate(is_val):
                if not is_valid:
                    continue
                if i >= len(x_cam_list) or i >= len(y_cam_list):
                    continue

                stem = f"{i:05d}"
                left_path = session_crops / f"{stem}_left.jpg"
                if not left_path.exists():
                    continue  # frame não processado pelo preprocess.py

                self.samples.append({
                    "session":    session_dir.name,
                    "stem":       stem,
                    "crop_dir":   session_crops,
                    "gaze_x_cm":  float(x_cam_list[i]),
                    "gaze_y_cm":  float(y_cam_list[i]),
                    "head_yaw":   0.0,   # gerado pelo preprocess.py — lido do metadata
                    "head_pitch": 0.0,
                    "head_roll":  0.0,
                    "frame_w":    640,
                    "frame_h":    480,
                    "face_bbox":  (0, 0, 640, 480),  # atualizado pelo preprocess.py
                })

        # Carregar head pose e face_bbox do metadata gerado pelo preprocess.py
        self._load_preprocess_metadata()

        print(f"GazeCaptureDataset [{split}]: {len(self.samples)} amostras "
              f"({len(sessions)} sessões)")

    def _load_preprocess_metadata(self) -> None:
        """
        Carrega head pose e face_bbox salvos pelo preprocess.py.

        O preprocess.py salva um metadata.json por sessão em crops/{session}/
        com os valores de yaw, pitch, roll e face_bbox por frame.
        """
        updated = []
        for s in self.samples:
            meta_path = s["crop_dir"] / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                frame_meta = meta.get(s["stem"], {})
                s["head_yaw"]   = frame_meta.get("yaw",   0.0)
                s["head_pitch"] = frame_meta.get("pitch", 0.0)
                s["head_roll"]  = frame_meta.get("roll",  0.0)
                s["face_bbox"]  = tuple(frame_meta.get("face_bbox", [0, 0, 640, 480]))
                s["frame_w"]    = frame_meta.get("frame_w", 640)
                s["frame_h"]    = frame_meta.get("frame_h", 480)
            updated.append(s)
        self.samples = updated

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        left, right, face = _load_crops(s["crop_dir"], s["stem"])
        left_t, right_t, face_t, grid_t = _crops_to_tensors(
            left, right, face, s["face_bbox"], s["frame_w"], s["frame_h"]
        )

        head_pose = torch.tensor(
            [s["head_yaw"], s["head_pitch"], s["head_roll"]],
            dtype=torch.float32,
        )
        gaze = torch.tensor(
            [s["gaze_x_cm"], s["gaze_y_cm"]],
            dtype=torch.float32,
        )

        return {
            "left_eye":  left_t,    # (3, 112, 112)
            "right_eye": right_t,   # (3, 112, 112)
            "face":      face_t,    # (3, 224, 224)
            "face_grid": grid_t,    # (1, 25, 25)
            "head_pose": head_pose, # (3,)
            "gaze":      gaze,      # (2,) em cm
        }


# ── ETH-XGaze (placeholder — integrar quando acesso chegar) ──────────────────

class ETHXGazeDataset(Dataset):
    """
    Placeholder para ETH-XGaze.

    Mesma interface do GazeCaptureDataset — drop-in replacement.
    Implementar quando o acesso ao dataset for aprovado.

    A pipeline de crops será idêntica: preprocess.py com MediaPipe,
    head pose via solvePnP, gaze em cm relativo à câmera.
    """

    def __init__(self, root: str | Path, split: str = "train", augment: bool = False) -> None:
        raise NotImplementedError(
            "ETHXGazeDataset será implementado quando o acesso for aprovado. "
            "Use GazeCaptureDataset por enquanto."
        )

    def __len__(self) -> int: return 0
    def __getitem__(self, idx: int) -> dict: ...


# ── MPIIFaceGaze ─────────────────────────────────────────────────────────────

class MPIIFaceGazeDataset(Dataset):
    """
    Lê datasets/MPIIFaceGaze/mpiifacegaze.csv gerado pelo prepare_mpiifacegaze.py.

    Cada sample retorna:
        left_eye:  (3, 112, 112) — normalizado ImageNet
        right_eye: (3, 112, 112) — normalizado ImageNet
        face:      (3, 224, 224) — normalizado ImageNet
        face_grid: (1, 25, 25)   — binário float32
        head_pose: (3,)          — [yaw, pitch, roll] já normalizados por π (lidos do CSV)
        gaze:      (2,)          — [pitch_norm, yaw_norm] já normalizados por π/2 (lidos do CSV)

    Split: últimos 2 participantes (p13, p14) para validação, restantes para treino.
    """

    _VAL_PARTICIPANTS = {"p13", "p14"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        self.root    = Path(root)
        self.augment = augment
        self.samples: list[dict] = []
        self._load(split)

    def _load(self, split: str) -> None:
        csv_path = self.root / "mpiifacegaze.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV não encontrado: {csv_path}\n"
                "Execute: python datasets/prepare_mpiifacegaze.py"
            )

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                is_val = row["participant"] in self._VAL_PARTICIPANTS
                if split == "val"   and not is_val: continue
                if split == "train" and     is_val: continue
                self.samples.append(row)

        print(f"MPIIFaceGazeDataset [{split}]: {len(self.samples)} amostras")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        crop_dir  = Path(s["crop_dir"])
        stem      = s["stem"]
        face_bbox = (
            int(s["face_bbox_x1"]), int(s["face_bbox_y1"]),
            int(s["face_bbox_x2"]), int(s["face_bbox_y2"]),
        )

        left, right, face = _load_crops(crop_dir, stem)
        left_t, right_t, face_t, grid_t = _crops_to_tensors(
            left, right, face, face_bbox,
            int(s["frame_w"]), int(s["frame_h"]),
        )

        head_pose = torch.tensor(
            [float(s["head_yaw"]), float(s["head_pitch"]), float(s["head_roll"])],
            dtype=torch.float32,
        )
        gaze = torch.tensor(
            [float(s["gaze_pitch"]), float(s["gaze_yaw"])],
            dtype=torch.float32,
        )

        return {
            "left_eye":  left_t,    # (3, 112, 112)
            "right_eye": right_t,   # (3, 112, 112)
            "face":      face_t,    # (3, 224, 224)
            "face_grid": grid_t,    # (1, 25, 25)
            "head_pose": head_pose, # (3,) — [yaw, pitch, roll] normalizados por π
            "gaze":      gaze,      # (2,) — [pitch_norm, yaw_norm] normalizados por π/2
        }


# ── Combinação de datasets ────────────────────────────────────────────────────

def build_dataset(
    gazecapture_root:  str | Path | None = None,
    ethxgaze_root:     str | Path | None = None,
    mpiifacegaze_root: str | Path | None = None,
    split: str = "train",
    augment: bool = False,
) -> Dataset:
    """
    Combina múltiplos datasets em um único Dataset PyTorch.

    Uso com MPIIFaceGaze:
        ds = build_dataset(mpiifacegaze_root="datasets/MPIIFaceGaze", split="train")

    Uso combinado:
        ds = build_dataset(
            mpiifacegaze_root="datasets/MPIIFaceGaze",
            gazecapture_root="datasets/GazeCapture",
            split="train",
        )
    """
    datasets = []

    if gazecapture_root:
        datasets.append(GazeCaptureDataset(gazecapture_root, split=split, augment=augment))

    if ethxgaze_root:
        datasets.append(ETHXGazeDataset(ethxgaze_root, split=split, augment=augment))

    if mpiifacegaze_root:
        datasets.append(MPIIFaceGazeDataset(mpiifacegaze_root, split=split, augment=augment))

    if not datasets:
        raise ValueError(
            "Passe ao menos um dataset: gazecapture_root, ethxgaze_root ou mpiifacegaze_root."
        )

    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    print(f"Dataset combinado [{split}]: {len(combined)} amostras total")
    return combined