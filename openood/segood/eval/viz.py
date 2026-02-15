import os
import numpy as np
import matplotlib.pyplot as plt

def save_overlay_triplet(image, mask, heatmap, out_path, title=""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = np.asarray(image)
    m = np.asarray(mask).astype(bool)
    h = np.asarray(heatmap)

    # normalize heatmap for display only
    h_disp = (h - np.min(h)) / (np.max(h) - np.min(h) + 1e-8)

    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.imshow(h_disp, alpha=0.45)
    plt.contour(m, levels=[0.5], linewidths=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
