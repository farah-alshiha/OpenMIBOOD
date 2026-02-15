import os
import argparse
import numpy as np
import torch

from openood.segood.models.unet2d import UNet2D
from openood.segood.methods.entropy import pixel_entropy
from openood.segood.methods.maxsoftmax import pixel_1_minus_maxsoftmax
from openood.segood.eval.viz import save_overlay_triplet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/segood/sanity", type=str)
    parser.add_argument("--H", default=256, type=int)
    parser.add_argument("--W", default=256, type=int)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Fake single sample (for now) -----
    # Replace this later with a real REMBRANDT T1 slice + tumor mask.
    img = np.random.randn(args.H, args.W).astype(np.float32)
    tumor_mask = (np.random.rand(args.H, args.W) > 0.97).astype(np.uint8)

    x = torch.from_numpy(img)[None, None].to(device)  # [B=1, C=1, H, W]

    # ----- Model -----
    model = UNet2D(in_channels=1, num_classes=2).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x)[0]  # [C, H, W]

    # ----- Pixel OOD scores -----
    ent = pixel_entropy(logits).cpu().numpy()                # [H, W]
    oms = pixel_1_minus_maxsoftmax(logits).cpu().numpy()     # [H, W]

    # ----- Save quick overlays -----
    save_overlay_triplet(
        image=img,
        mask=tumor_mask,
        heatmap=ent,
        out_path=os.path.join(args.out_dir, "overlay_entropy.png"),
        title="Sanity: entropy heatmap (random weights)"
    )

    save_overlay_triplet(
        image=img,
        mask=tumor_mask,
        heatmap=oms,
        out_path=os.path.join(args.out_dir, "overlay_1minusmax.png"),
        title="Sanity: 1-maxsoftmax heatmap (random weights)"
    )

    print(f"Saved sanity overlays to: {args.out_dir}")


if __name__ == "__main__":
    main()
