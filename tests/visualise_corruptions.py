"""Visual verification of the corruption pipeline.

Applies all 23 corruption conditions to a synthetic test image and saves a
labelled PNG grid for human inspection.

Usage:
    python tests/visualise_corruptions.py

Output:
    tests/corruption_grid.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corruption.pipeline import apply_corruption
from src.corruption.params import RGB_CORRUPTIONS, TIR_CORRUPTIONS

# ── Build a synthetic test image with visible structure ───────────────────────

def make_test_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Gradient + circle pattern so blur/noise/contrast are all visible."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal gradient
    for x in range(w):
        img[:, x, 0] = int(x / w * 200) + 30
    # Vertical gradient on green channel
    for y in range(h):
        img[y, :, 1] = int(y / h * 200) + 30
    # Blue circle
    cx, cy, r = w // 2, h // 2, h // 4
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask, 2] = 200
    return img

# ── Layout definition ─────────────────────────────────────────────────────────

RGB_TYPES = ["gaussian_noise", "motion_blur", "brightness_shift", "low_contrast", "complete_dropout"]
TIR_TYPES = ["sensor_noise", "blur", "intensity_shift", "complete_dropout"]

# Columns: sev1 | sev2 | sev3 | dropout (binary types span all columns)
SEVERITIES = [1, 2, 3]

def get_conditions(corruption_types, corruption_table):
    """Return list of (label, severity_or_None) for each cell in the grid row."""
    rows = []
    for ctype in corruption_types:
        if corruption_table[ctype] is None:
            rows.append((ctype, [None, None, None]))  # dropout spans 3 cols
        else:
            rows.append((ctype, [1, 2, 3]))
    return rows

# ── Render ────────────────────────────────────────────────────────────────────

def render_grid():
    np.random.seed(42)

    rgb_base = make_test_image()
    tir_base = (0.299 * rgb_base[:, :, 0] +
                0.587 * rgb_base[:, :, 1] +
                0.114 * rgb_base[:, :, 2]).astype(np.uint8)

    rgb_rows = get_conditions(RGB_TYPES, RGB_CORRUPTIONS)
    tir_rows = get_conditions(TIR_TYPES, TIR_CORRUPTIONS)

    n_cols = 4  # sev1, sev2, sev3, label col → actually 3 image cols + row label
    n_rows = 1 + len(rgb_rows) + 1 + len(tir_rows)  # header + rgb + divider + tir

    fig_w = 3 * 3 + 2   # 3 image cols × 3 inches + label space
    fig_h = n_rows * 2.2

    fig, axes = plt.subplots(n_rows, 4, figsize=(fig_w, fig_h),
                              gridspec_kw={"width_ratios": [1.8, 1, 1, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    def hide(ax):
        ax.set_visible(False)

    def label_cell(ax, text, color="#e0e0e0", size=9, bold=False):
        ax.set_facecolor("#1a1a2e")
        ax.axis("off")
        weight = "bold" if bold else "normal"
        ax.text(0.5, 0.5, text, ha="center", va="center",
                color=color, fontsize=size, fontweight=weight,
                transform=ax.transAxes)

    def img_cell(ax, image, title=None):
        ax.set_facecolor("#0f0f1a")
        ax.axis("off")
        if image.ndim == 2:
            ax.imshow(image, cmap="inferno", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        if title:
            ax.set_title(title, color="#aaaaaa", fontsize=7, pad=2)

    # Row 0: column headers
    label_cell(axes[0, 0], "Corruption", color="#ffffff", size=10, bold=True)
    label_cell(axes[0, 1], "Severity 1", color="#7ec8e3", size=9, bold=True)
    label_cell(axes[0, 2], "Severity 2", color="#7ec8e3", size=9, bold=True)
    label_cell(axes[0, 3], "Severity 3", color="#7ec8e3", size=9, bold=True)

    row_idx = 1

    def render_rows(corruption_rows, modality, base_image):
        nonlocal row_idx
        for cname, severities in corruption_rows:
            label = cname.replace("_", "\n")
            mod_color = "#f4a261" if modality == "rgb" else "#2a9d8f"
            label_cell(axes[row_idx, 0], f"[{modality.upper()}]\n{label}",
                       color=mod_color, size=8)
            for col, sev in enumerate(severities):
                np.random.seed(42)
                corrupted = apply_corruption(base_image, modality, cname, sev)
                title = "dropout" if sev is None else f"sev {sev}"
                img_cell(axes[row_idx, col + 1], corrupted, title)
            row_idx += 1

    render_rows(rgb_rows, "rgb", rgb_base)

    # Divider row
    for col in range(4):
        axes[row_idx, col].set_facecolor("#2d2d4e")
        axes[row_idx, col].axis("off")
        if col == 0:
            axes[row_idx, col].text(0.5, 0.5, "── TIR ──", ha="center", va="center",
                                    color="#ffffff", fontsize=9, fontweight="bold",
                                    transform=axes[row_idx, col].transAxes)
    row_idx += 1

    render_rows(tir_rows, "tir", tir_base)

    plt.suptitle("Corruption Pipeline — All 23 Conditions", color="#ffffff",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.4)

    out_path = Path(__file__).parent / "corruption_grid.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    path = render_grid()
    print(f"Open {path} to inspect all 23 corruption conditions.")
