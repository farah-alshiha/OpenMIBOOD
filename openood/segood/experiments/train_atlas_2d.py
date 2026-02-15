import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.segood.data.atlas import Atlas2DConfig, AtlasLesion2DSlices
from openood.segood.models.unet2d import UNet2D


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred/target: [B,H,W] 0/1
    inter = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    return (2 * inter + eps) / (union + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_meta", required=True)
    ap.add_argument("--atlas_split", required=True)
    ap.add_argument("--out_dir", default="results/segood/atlas_unet2d")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train_ds = AtlasLesion2DSlices(Atlas2DConfig(args.atlas_meta, args.atlas_split, "train"))
    val_ds   = AtlasLesion2DSlices(Atlas2DConfig(args.atlas_meta, args.atlas_split, "val"))

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = UNet2D(in_channels=1, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # class imbalance: lesion is rare, weight lesion higher
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {ep:02d} [Train]", leave=False)
        for x, y, _ in train_bar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())


        # ---- val ----
        model.eval()
        dices = []
        val_bar = tqdm(val_loader, desc=f"Epoch {ep:02d} [Val]", leave=False)
        with torch.no_grad():
            for x, y, _ in val_bar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                pred = logits.argmax(dim=1)

                d = dice_score((pred == 1).float(), (y == 1).float())
                dices.append(d.cpu())

                val_bar.set_postfix(batch_dice=d.mean().item())


        mean_dice = torch.cat(dices).mean().item()
        print(f"Epoch {ep:02d} | train_loss={total_loss/len(train_loader):.4f} | val_dice={mean_dice:.4f}")

        if mean_dice > best_val:
            best_val = mean_dice
            torch.save({"model": model.state_dict(), "val_dice": best_val}, best_path)

    print(f"Epoch {ep:02d} | train_loss={total_loss/len(train_loader):.4f} | val_dice={mean_dice:.4f}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()