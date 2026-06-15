# train.py — IrisFlow 2.0
"""
Treino do IrisGazeNet v3.

Uso:
    python core/train.py --mpiifacegaze datasets/MPIIFaceGaze --epochs 30 --out models/irisgazenet_v3.pt

Para fine-tunar os backbones também: --finetune-backbones
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import build_dataset
from model import IrisGazeNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    # Backbones em eval se não estiverem sendo fine-tunados
    # (preserva BatchNorm estatísticas do ImageNet)
    losses = []
    for batch in loader:
        left      = batch["left_eye"].to(device)
        right     = batch["right_eye"].to(device)
        face      = batch["face"].to(device)
        grid      = batch["face_grid"].to(device)
        pose      = batch["head_pose"].to(device)
        gaze      = batch["gaze"].to(device)

        pred = model(left, right, face, grid, pose)
        loss = nn.MSELoss()(pred, gaze)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    errors = []
    for batch in loader:
        left  = batch["left_eye"].to(device)
        right = batch["right_eye"].to(device)
        face  = batch["face"].to(device)
        grid  = batch["face_grid"].to(device)
        pose  = batch["head_pose"].to(device)
        gaze  = batch["gaze"].to(device)

        pred = model(left, right, face, grid, pose)

        # Erro euclidiano em espaço normalizado (pitch/yaw)
        err = torch.sqrt(((pred - gaze)**2).sum(dim=1))
        errors.extend(err.cpu().tolist())

    errors = np.array(errors)
    return {
        "mae_norm":    float(np.mean(errors)),
        "median_norm": float(np.median(errors)),
        "p90_norm":    float(np.percentile(errors, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpiifacegaze",        default="datasets/MPIIFaceGaze")
    parser.add_argument("--gazecapture",         default=None)
    parser.add_argument("--ethxgaze",            default=None)
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--batch-size",          type=int,   default=64)
    parser.add_argument("--workers",             type=int,   default=4)
    parser.add_argument("--out",                 default="models/irisgazenet_v3.pt")
    parser.add_argument("--finetune-backbones",  action="store_true",
                        help="Fine-tunar os 3 backbones além do MLP")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_ds = build_dataset(
        mpiifacegaze_root=args.mpiifacegaze,
        gazecapture_root=args.gazecapture,
        ethxgaze_root=args.ethxgaze,
        split="train",
        augment=True,
    )
    val_ds = build_dataset(
        mpiifacegaze_root=args.mpiifacegaze,
        gazecapture_root=args.gazecapture,
        ethxgaze_root=args.ethxgaze,
        split="val",
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = IrisGazeNet().to(device)

    # Parâmetros treináveis
    if args.finetune_backbones:
        params = model.parameters()
        logger.info("Fine-tunando backbones + MLP")
    else:
        # Congela os 3 backbones — treina só o MLP
        for net in [model.left_eye_net, model.right_eye_net, model.face_net]:
            for p in net.parameters():
                p.requires_grad = False
        params = model.mlp.parameters()
        logger.info("Treinando apenas o MLP (backbones congelados)")

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_m = train_epoch(model, train_loader, optimizer, device)
        val_m   = evaluate(model, val_loader, device)
        scheduler.step()

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | val_mae=%.4f | val_p90=%.4f | %.1fs",
            epoch, args.epochs,
            train_m["loss"], val_m["mae_norm"], val_m["p90_norm"],
            time.perf_counter() - t0,
        )

        if val_m["mae_norm"] < best_mae:
            best_mae = val_m["mae_norm"]
            model.save(args.out)
            logger.info("  ✓ Modelo salvo (MAE=%.4f)", best_mae)

    logger.info("Treino concluído — melhor MAE: %.4f", best_mae)


if __name__ == "__main__":
    main()
