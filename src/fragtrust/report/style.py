"""Figure style for an LNCS page: 122 mm text width, 7-8 pt type, greyscale-safe series.

Series colours are the first three slots of a palette validated for colour-vision deficiency
(all pairs); every series also gets its own marker and line style, so figures survive greyscale
printing. More than three base models per figure are refused rather than given a fourth hue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH = 122 / 25.4

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EMPTY = "#f3f2ee"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def model_styles(models: list[str]) -> dict[str, dict]:
    """Colour, marker and line style per model; an ensemble `<base>_ens` reuses its base model's
    colour with a dashed line and open markers."""
    bases = [m for m in models if not (m.endswith("_ens") and m[:-4] in models)]
    if len(bases) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} base models per figure, got {bases}")
    styles = {m: {"color": SERIES[i], "marker": MARKERS[i], "linestyle": "-", "markerfacecolor": SERIES[i]} for i, m in enumerate(bases)}
    for m in models:
        if m not in styles:
            styles[m] = {**styles[m[:-4]], "linestyle": "--", "markerfacecolor": "white"}
    return styles


def save(fig, out_dir: Path, name: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.svg", out_dir / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=600)
    plt.close(fig)
    return [str(p) for p in paths]
