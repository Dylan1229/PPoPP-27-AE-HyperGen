"""The RICE palette and the figure style used by the paper's figures.

The paper's figures were drawn with an in-house plotting module (`scientific_figure_pro.py`). If a
copy of it sits next to this file, it is used verbatim. Otherwise the fallback below provides the
same palette and style, so every figure can be regenerated anywhere.

The colour values are the ones documented for each role in the paper's plotting scripts. Anything
that needs a value not recorded there raises rather than guessing, because a figure in a
nearly-right palette is worse than one that failed to build: it looks finished.

Usage:

    from rice_palette import sfp, RICE
    sfp.apply_publication_style(sfp.FigureStyle(font_size=13, axes_linewidth=1.2))
    ...
    sfp.finalize_figure(fig, path, formats=["pdf"], close=False, pad=0.04)
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- the real module, if present
_CANDIDATES = [
    Path(__file__).resolve().parent / "scientific_figure_pro.py",
]

sfp = None
for _p in _CANDIDATES:
    if _p.is_file():
        _spec = importlib.util.spec_from_file_location("sfp", _p)
        sfp = importlib.util.module_from_spec(_spec)
        sys.modules["sfp"] = sfp
        _spec.loader.exec_module(sfp)
        break

if sfp is not None:
    RICE = sfp.RICE
    USING_FALLBACK = False
else:
    # ------------------------------------------------------------ fallback
    # Hexes transcribed from replot_eval_rice.py:38-42, where each role is named beside its
    # value. Only these five are known; a KeyError on anything else is deliberate.
    RICE = {
        "ours": "#0A509E",            # deep blue   — our full system
        "best_baseline": "#E0A494",   # salmon      — parallelism only
        "other_baseline": "#7C7E7F",  # gray        — origin / without
        "ablation": "#C04829",        # brick red   — second model
        "highlight": "#E9A139",       # amber       — speedup callouts
    }
    USING_FALLBACK = True

    import matplotlib
    import matplotlib.pyplot as plt

    @dataclass
    class FigureStyle:
        font_size: int = 13
        axes_linewidth: float = 1.2

    def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
        """Match the original's typography closely enough to sit beside its output.

        Serif at the given size with bold labels, black axes at the given width, no top or
        right spine. Not a claim of byte-identity with the real module -- see the note printed
        at import time.
        """
        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "font.size": style.font_size,
            "axes.labelsize": style.font_size,
            "axes.titlesize": style.font_size + 1,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": style.axes_linewidth,
            "axes.edgecolor": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": style.font_size - 1,
            "ytick.labelsize": style.font_size - 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": style.font_size - 2,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,     # embed TrueType so the PDF is editable and text-searchable
            "ps.fonttype": 42,
        })

    def finalize_figure(fig, path, formats=("pdf",), dpi=None, close=True, pad=0.04):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out = path if path.suffix == f".{fmt}" else path.with_suffix(f".{fmt}")
            fig.savefig(out, format=fmt, bbox_inches="tight", pad_inches=pad,
                        **({"dpi": dpi} if dpi else {}))
        if close:
            plt.close(fig)

    _mod = sys.modules[__name__]
    _mod.FigureStyle = FigureStyle                      # type: ignore[attr-defined]
    _mod.apply_publication_style = apply_publication_style   # type: ignore[attr-defined]
    _mod.finalize_figure = finalize_figure              # type: ignore[attr-defined]
    sfp = _mod
    # The paper's figures were drawn with an in-house style module; this fallback reproduces its
    # colours exactly and its typography closely. Data and values are unaffected.
