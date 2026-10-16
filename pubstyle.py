"""
pubstyle -- a muted, colorblind-aware, minimalist figure style for publications.

Usage
-----
    import pubstyle as ps
    ps.use()                              # matplotlib + seaborn, mathtext
    # ps.use(latex=True)                  # real LaTeX text rendering (needs a TeX install)

    fig, ax = plt.subplots(figsize=ps.figsize("single"))
    ax.plot(x, y, color=ps.C["indigo"])
    ps.save(fig, "figure1")               # -> figure1.pdf (vector) + figure1.png (300 dpi)

Design rules
------------
* Categorical colors: 7 muted hues + a neutral grey, in a FIXED order. Neighbouring colors
  stay distinguishable under deuteranopia / protanopia / tritanopia (checked in OKLab,
  Machado et al. 2009 simulation). The first four colors are also safe in any combination
  (scatter plots, small multiples), so use them first.
* Sequential (magnitude): one hue (indigo), light -> dark, monotone lightness.
* Diverging (polarity): indigo <- neutral -> clay, equal lightness steps per arm.
* Because several muted tones have modest contrast against white, important distinctions
  should ALSO carry a second cue (direct labels, markers, linestyles, hatching).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_hex

__all__ = [
    "C", "CATEGORICAL", "GREY", "SEQ", "DIV", "use", "figsize", "save",
    "panel_labels", "tint", "cvd", "palette_hex", "export_colors", "write_mplstyle",
    "MARKERS", "LINESTYLES", "HATCHES",
]

# --------------------------------------------------------------------------------------
# 1. Colors
# --------------------------------------------------------------------------------------
# Fixed order. Do not shuffle: the order is what keeps neighbours CVD-distinguishable.
C = {
    "indigo":     "#525A94",   # 1  deep, calm blue-violet  (primary / "ours")
    "clay":       "#D68A4E",   # 2  warm orange             (main contrast to 1)
    "sage":       "#50B095",   # 3  muted teal-green
    "rose":       "#B8566B",   # 4  dusty rose / wine
    "periwinkle": "#848DC8",   # 5  light blue-violet
    "ochre":      "#C9A53F",   # 6  mustard
    "sky":        "#6DBBE4",   # 7  pale sky blue
    "grey":       "#868686",   # 8  neutral: baselines, "other", reference data
}
CATEGORICAL = [C[k] for k in ("indigo", "clay", "sage", "rose", "periwinkle", "ochre", "sky")]
GREY = C["grey"]
INK = "#2B2B2B"          # text, axes
INK_LIGHT = "#6B6B6B"    # secondary text, minor elements
GRID = "#E6E6E6"         # optional light grid

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]            # secondary encoding for lines/points
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
HATCHES = ["", "///", "\\\\\\", "...", "xxx", "|||", "---"]  # secondary encoding for fills


def tint(color, amount=0.75):
    """Blend a color toward white (amount=0 -> original, 1 -> white). Good for fills/bands."""
    rgb = np.array(mpl.colors.to_rgb(color))
    return to_hex(rgb + (1 - rgb) * amount)


# ---- OKLCH helpers (to build perceptually even sequential/diverging ramps) -------------
def _lin2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def _oklch_to_rgb(L, Ch, h_deg):
    """OKLCH -> sRGB in [0,1]; chroma is reduced until the color is inside the gamut."""
    for scale in np.linspace(1, 0, 60):
        a = Ch * scale * math.cos(math.radians(h_deg))
        b = Ch * scale * math.sin(math.radians(h_deg))
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
        rgb = np.array([
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ])
        if rgb.min() >= -1e-4 and rgb.max() <= 1 + 1e-4:
            break
    return _lin2s(rgb)


def _hue_of(hex_color):
    r, g, b = mpl.colors.to_rgb(hex_color)
    lin = np.array([((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in (r, g, b)])
    l = 0.4122214708 * lin[0] + 0.5363325363 * lin[1] + 0.0514459929 * lin[2]
    m = 0.2119034982 * lin[0] + 0.6806995451 * lin[1] + 0.1073969566 * lin[2]
    s = 0.0883024619 * lin[0] + 0.2817188376 * lin[1] + 0.6299787005 * lin[2]
    l, m, s = np.cbrt([l, m, s])
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return math.degrees(math.atan2(b, a)) % 360


_H_COOL = _hue_of(C["indigo"])
_H_WARM = _hue_of(C["clay"])


def _seq_colors(n=256):
    out = []
    for t in np.linspace(0, 1, n):
        L = 0.975 - 0.68 * t                       # 0.975 -> 0.295, strictly monotone
        Ch = 0.012 + 0.072 * math.sin(math.pi * (0.10 + 0.80 * t))
        out.append(_oklch_to_rgb(L, Ch, _H_COOL))
    return np.array(out)


def _div_colors(n=257):
    half = n // 2
    arm = np.linspace(0, 1, half + 1)               # 0 = neutral, 1 = end

    def make(h, c_end):
        cols = []
        for t in arm:
            L = 0.955 - 0.57 * t                    # same lightness profile on both arms
            Ch = c_end * (math.sin(math.pi / 2 * t) ** 0.9) * (1 - 0.25 * t ** 3)
            cols.append(_oklch_to_rgb(L, Ch, h))
        return np.array(cols)

    cool, warm = make(_H_COOL, 0.105), make(_H_WARM, 0.105)
    mid = _oklch_to_rgb(0.955, 0.0, 0.0)            # neutral grey midpoint (no hue)
    return np.vstack([cool[::-1][:-1], mid[None, :], warm[1:]])


SEQ = ListedColormap(_seq_colors(), name="pub_seq")
DIV = ListedColormap(_div_colors(), name="pub_div")
CAT = ListedColormap(CATEGORICAL, name="pub_cat")
for _cm in (SEQ, DIV, CAT):
    for _c in (_cm, _cm.reversed()):
        try:
            mpl.colormaps.register(_c, name=_c.name, force=True)
        except Exception:  # very old matplotlib
            pass


def palette_hex(n=None):
    """Categorical hex list; n<=7 returns the first n colors (never cycled)."""
    if n is not None and n > len(CATEGORICAL):
        raise ValueError(
            f"Only {len(CATEGORICAL)} categorical colors; fold extras into 'Other' ({GREY}), "
            "use small multiples, or add markers/linestyles instead of more hues.")
    return CATEGORICAL[:n] if n else list(CATEGORICAL)


# ---- CVD simulation (Machado, Oliveira & Fernandes 2009, severity 1.0) ------------------
_MACHADO = {
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}


def cvd(colors, kind="deutan"):
    """Simulate how colors look with protan/deutan/tritan CVD ('gray' for grayscale print)."""
    rgb = np.array([mpl.colors.to_rgb(c) for c in colors])
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    if kind == "gray":
        y = lin @ np.array([0.2126, 0.7152, 0.0722])
        lin = np.stack([y, y, y], axis=1)
    else:
        lin = np.clip(lin @ _MACHADO[kind].T, 0, 1)
    return [to_hex(c) for c in _lin2s(lin)]


# --------------------------------------------------------------------------------------
# 2. Style (rcParams)
# --------------------------------------------------------------------------------------
FONT_SANS = ["Arial", "Helvetica", "Liberation Sans", "FreeSans", "Inter", "DejaVu Sans"]
FONT_SERIF = ["Times New Roman", "Liberation Serif", "FreeSerif", "DejaVu Serif"]

# page geometry (mm). Defaults follow Nature / Science style single & double column.
COLUMN_MM = {"single": 89, "onehalf": 120, "double": 183}


def figsize(width="single", aspect=0.72, height=None):
    """Figure size in inches. width: 'single' | 'onehalf' | 'double' | millimetres | None.
    aspect = height/width unless an explicit height (mm) is given."""
    w_mm = COLUMN_MM[width] if isinstance(width, str) else float(width)
    h_mm = float(height) if height is not None else w_mm * aspect
    return (w_mm / 25.4, h_mm / 25.4)


def _rc(fontsize=7.5, redundant=False):
    cyc = cycler(color=CATEGORICAL)
    if redundant:
        cyc = cyc + cycler(marker=MARKERS)
    return {
        # --- fonts ---
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS,
        "font.serif": FONT_SERIF,
        "font.size": fontsize,
        "axes.titlesize": fontsize + 0.5,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "legend.fontsize": fontsize - 0.5,
        "figure.titlesize": fontsize + 1,
        "mathtext.fontset": "stixsans",   # sans-serif math, upright digits, italic variables
        # --- colors ---
        "axes.prop_cycle": cyc,
        "image.cmap": "pub_seq",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        # --- minimal axes ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.labelpad": 2.5,
        "axes.autolimit_mode": "data",
        "axes.xmargin": 0.04,
        "axes.ymargin": 0.04,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        # --- marks ---
        "lines.linewidth": 1.3,
        "lines.markersize": 3.8,
        "lines.markeredgewidth": 0.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "patch.edgecolor": "white",
        "scatter.edgecolors": "none",
        "errorbar.capsize": 1.8,
        "boxplot.flierprops.markersize": 2,
        "hist.bins": "auto",
        # --- legend ---
        "legend.frameon": False,
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.2,
        # --- layout / output ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.h_pad": 0.03,
        "pdf.fonttype": 42,        # TrueType, text stays text/selectable, accepted by journals
        "ps.fonttype": 42,
        "svg.fonttype": "path",    # glyphs as outlines -> same look in Typst / Inkscape / browsers
    }


_LATEX_PREAMBLE = {
    "sans": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{helvet}"
            r"\usepackage{sansmath}\renewcommand{\familydefault}{\sfdefault}\sansmath",
    "serif": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{newtxtext,newtxmath}",
}


def use(latex=False, latex_font="sans", fontsize=7.5, redundant=False, seaborn=True, **overrides):
    """Activate the style globally for matplotlib (and seaborn, if installed).

    latex=True       render all text with a real TeX install (match your paper's fonts;
                     tweak ps._LATEX_PREAMBLE or pass text.latex.preamble=... to override).
    latex_font       'sans' (Helvetica-like, default) or 'serif' for the TeX text/math font.
    fontsize         base font size in pt (7-8 pt is typical for journal figures at final size).
    redundant=True   also cycle markers in the property cycle (secondary, non-color encoding).
    """
    rc = _rc(fontsize, redundant)
    if seaborn:
        try:
            import seaborn as sns
            sns.set_theme(context="paper", style="ticks", palette=CATEGORICAL)
        except ImportError:
            pass
    mpl.rcParams.update(rc)
    if latex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif" if latex_font == "sans" else "serif",
            "text.latex.preamble": _LATEX_PREAMBLE[latex_font],
        })
    else:
        mpl.rcParams["text.usetex"] = False
    mpl.rcParams.update(overrides)
    return mpl.rcParams


def write_mplstyle(path="pubstyle.mplstyle"):
    """Write the rcParams as a standalone .mplstyle (for plt.style.use('pubstyle.mplstyle'))."""
    rc = _rc()
    lines = ["# pubstyle: muted, colorblind-aware, minimalist (generated by pubstyle.write_mplstyle)"]
    for k, v in rc.items():
        if k == "axes.prop_cycle":
            v = "cycler('color', [%s])" % ", ".join(repr(c.lstrip("#")) for c in CATEGORICAL)
        elif isinstance(v, list):
            v = ", ".join(map(str, v))
        elif isinstance(v, bool):
            v = str(v)
        elif isinstance(v, str) and v.startswith("#"):
            v = v.lstrip("#")           # .mplstyle treats '#' as a comment
        lines.append(f"{k}: {v}")
    Path(path).write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------
# 3. Helpers
# --------------------------------------------------------------------------------------
def panel_labels(axes, labels=None, fmt="{}", dx=-0.02, dy=1.04, size=None, weight="bold"):
    """Add a, b, c ... panel labels at the top-left of each axes (axes coordinates)."""
    axes = np.ravel(axes)
    labels = labels or [chr(ord("a") + i) for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.text(dx, dy, fmt.format(lab), transform=ax.transAxes, fontweight=weight,
                fontsize=size or mpl.rcParams["font.size"] + 1.5, va="bottom", ha="right")


def save(fig, name, formats=("pdf", "png"), **kwargs):
    """Save a figure in several formats (vector PDF for the paper, PNG for slides/review)."""
    paths = []
    for ext in formats:
        p = f"{name}.{ext}"
        fig.savefig(p, **kwargs)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------------------
# 4. Export to LaTeX / Typst
# --------------------------------------------------------------------------------------
def _stops(cmap, n=9):
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def export_colors(outdir="."):
    """Write palette.tex (xcolor + pgfplots colormaps) and palette.typ (colors + gradients)."""
    out = Path(outdir)
    seq, div = _stops(SEQ), _stops(DIV)

    tex = ["% pubstyle colors -- \\input{palette.tex} after \\usepackage{xcolor} (and pgfplots)",
           "% Usage: \\textcolor{pubIndigo}{text}, color=pubClay in TikZ/pgfplots, colormap name=pubseq"]
    for name, hx in C.items():
        tex.append(f"\\definecolor{{pub{name.capitalize()}}}{{HTML}}{{{hx.lstrip('#')}}}")
    tex.append(f"\\definecolor{{pubInk}}{{HTML}}{{{INK.lstrip('#')}}}")
    tex.append("% pgfplots cycle list (fixed order) and colormaps")
    tex.append("\\ifdefined\\pgfplotsset")
    tex.append("\\pgfplotscreateplotcyclelist{pubcycle}{%")
    for name in ("indigo", "clay", "sage", "rose", "periwinkle", "ochre", "sky"):
        tex.append(f"  {{pub{name.capitalize()},mark=*}},%")
    tex.append("}")

    def pgf_map(nm, stops):
        pts = " ".join("rgb255=(%d,%d,%d)" % tuple(int(round(x * 255)) for x in mpl.colors.to_rgb(s))
                       for s in stops)
        return f"\\pgfplotsset{{colormap={{{nm}}}{{{pts}}}}}"

    tex += [pgf_map("pubseq", seq), pgf_map("pubdiv", div), "\\fi", ""]
    (out / "palette.tex").write_text("\n".join(tex))

    typ = ["// pubstyle colors -- #import \"palette.typ\": *", ""]
    for name, hx in C.items():
        typ.append(f'#let pub-{name} = rgb("{hx}")')
    typ.append(f'#let pub-ink = rgb("{INK}")')
    typ.append("#let pub-cycle = (" + ", ".join(f"pub-{n}" for n in
               ("indigo", "clay", "sage", "rose", "periwinkle", "ochre", "sky")) + ")")
    typ.append("#let pub-seq = gradient.linear(" + ", ".join(f'rgb("{s}")' for s in seq) + ")")
    typ.append("#let pub-div = gradient.linear(" + ", ".join(f'rgb("{s}")' for s in div) + ")")
    typ.append("")
    (out / "palette.typ").write_text("\n".join(typ))
    return [out / "palette.tex", out / "palette.typ"]


if __name__ == "__main__":
    write_mplstyle()
    export_colors()
    print("wrote pubstyle.mplstyle, palette.tex, palette.typ")
