"""
Example figures for the pubstyle palette. Run:  python examples.py [--latex]

Writes (PDF + PNG each): fig_palette, fig_matplotlib, fig_seaborn, fig_single
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

import pubstyle as ps

LATEX = "--latex" in sys.argv
rng = np.random.default_rng(42)
C = ps.C


# --------------------------------------------------------------------------------------
# Figure 0: the palette itself, incl. colorblind and grayscale simulations
# --------------------------------------------------------------------------------------
def fig_palette():
    names = list(C)
    cols = [C[n] for n in names]
    rows = [("normal vision", cols)] + [(lab, ps.cvd(cols, k)) for lab, k in
            (("deuteranopia", "deutan"), ("protanopia", "protan"),
             ("tritanopia", "tritan"), ("grayscale", "gray"))]
    fig = plt.figure(figsize=ps.figsize("double", height=88))
    gs = fig.add_gridspec(3, 1, height_ratios=[5.2, 1.0, 1.0], hspace=0.15)

    # categorical swatches + simulations
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-1.9, len(cols)); ax.set_ylim(-0.2, len(rows) + 0.15)
    ax.invert_yaxis(); ax.axis("off")
    for r, (lab, row) in enumerate(rows):
        ax.text(-0.1, r + 0.5, lab, ha="right", va="center", fontsize=7.5,
                fontweight="bold" if r == 0 else "regular",
                color=ps.INK if r == 0 else ps.INK_LIGHT)
        for i, c in enumerate(row):
            ax.add_patch(Rectangle((i + 0.04, r + 0.06), 0.92, 0.86, color=c, lw=0))
    for i, n in enumerate(names):
        ax.text(i + 0.5, len(rows) + 0.02, f"{i + 1}  {n}\n{C[n]}", ha="center", va="top",
                fontsize=6.5, color=ps.INK_LIGHT, linespacing=1.4)
    ax.set_title("Categorical: fixed order, never cycled (grey = other / reference)",
                 loc="left", fontsize=8)
    # sequential & diverging
    for k, (cmap, title) in enumerate(((ps.SEQ, "Sequential (magnitude): one hue, light to dark"),
                                       (ps.DIV, "Diverging (polarity): two hues, neutral midpoint"))):
        a = fig.add_subplot(gs[k + 1])
        a.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
        a.set_yticks([]); a.set_xticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(title, loc="left", fontsize=8)
    return fig


# --------------------------------------------------------------------------------------
# Figure 1: matplotlib, double column, six panels
# --------------------------------------------------------------------------------------
def fig_matplotlib():
    fig, axs = plt.subplots(2, 3, figsize=ps.figsize("double", height=102))
    a, b, c, d, e, f = axs.ravel()

    # (a) lines + uncertainty bands, direct labels instead of a legend
    t = np.linspace(0, 10, 200)
    series = {"Control": (np.exp(-t / 12), C["grey"], 0.0),
              "Drug A": (np.exp(-t / 5) + 0.05 * np.sin(t / 1.5), C["indigo"], 0.0),
              "Drug B": (1.15 * np.exp(-t / 2.2), C["clay"], -0.02)}
    for label, (y, col, dy) in series.items():
        a.fill_between(t, y - 0.07, y + 0.07, color=ps.tint(col, 0.7), lw=0)
        a.plot(t, y, color=col)
        a.text(10.2, y[-1] + dy, label, color=col, va="center", fontsize=7, fontweight="bold")
    a.set(xlabel="Time $t$ (s)", ylabel="Response (a.u.)", xlim=(0, 10), ylim=(0, 1.3))
    a.set_xlim(0, 10)
    a.spines["right"].set_visible(False)
    a.margins(x=0)
    a.set_clip_on(False)

    # (b) grouped bars: thin, white gaps, error bars
    groups = ["Low", "Mid", "High"]
    conds = ["WT", "KO", "Rescue"]
    means = np.array([[1.0, 1.9, 3.1], [1.1, 1.0, 1.2], [1.0, 1.6, 2.7]])
    errs = np.array([[0.15, 0.2, 0.3], [0.15, 0.15, 0.2], [0.2, 0.25, 0.3]])
    w = 0.26
    for i, cond in enumerate(conds):
        x = np.arange(3) + (i - 1) * (w + 0.03)
        b.bar(x, means[i], w, yerr=errs[i], color=ps.palette_hex(3)[i], label=cond,
              error_kw=dict(lw=0.7, capthick=0.7, ecolor=ps.INK))
    b.set_xticks(range(3), groups)
    b.set(ylabel="Expression (fold)", xlabel="Dose")
    b.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, 1.12), columnspacing=1, handlelength=0.9)
    b.set_ylim(0, 4)

    # (c) scatter, color + marker (secondary encoding)
    for i, (mu, lab) in enumerate((((0, 0), "A"), ((2.4, 1.2), "B"), ((0.8, 2.8), "C"), ((3, -1.2), "D"))):
        pts = rng.multivariate_normal(mu, [[0.35, 0.1 * (-1) ** i], [0.1 * (-1) ** i, 0.3]], 45)
        c.scatter(*pts.T, s=11, color=ps.CATEGORICAL[i], marker=ps.MARKERS[i], label=f"Cluster {lab}", alpha=0.9)
    c.set(xlabel="PC 1", ylabel="PC 2")
    c.legend(loc="upper right", markerscale=1.1, handletextpad=0.1, ncols=2, columnspacing=0.8,
             bbox_to_anchor=(1.02, 1.17))

    # (d) sequential heatmap
    xx, yy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    z = np.exp(-((xx - 0.8) ** 2 + (yy + 0.4) ** 2) / 2.2) + 0.55 * np.exp(-((xx + 1.5) ** 2 + (yy - 1.2) ** 2) / 1.1)
    im = d.imshow(z, extent=(-3, 3, -3, 3), origin="lower", cmap=ps.SEQ)
    d.set(xlabel="$x$ (mm)", ylabel="$y$ (mm)")
    for s in d.spines.values():
        s.set_visible(True)
    cb = fig.colorbar(im, ax=d, shrink=0.9, aspect=18, pad=0.02)
    cb.set_label("Intensity"); cb.outline.set_visible(False); cb.ax.tick_params(length=0)

    # (e) diverging heatmap (correlation), centred on zero
    n = 6                                          # latent factors with mixed-sign loadings
    load = np.array([[0.9, 0.1], [0.8, -0.2], [0.3, 0.8], [-0.2, 0.9], [-0.7, -0.4], [-0.6, 0.5]])
    corr = np.corrcoef(rng.normal(size=(300, 2)) @ load.T + rng.normal(0, 0.55, (300, n)), rowvar=False)
    im2 = e.imshow(corr, cmap=ps.DIV, vmin=-1, vmax=1)
    e.set_xticks(range(n), list("ABCDEF")); e.set_yticks(range(n), list("ABCDEF"))
    e.tick_params(length=0)
    for s in e.spines.values():
        s.set_visible(False)
    cb2 = fig.colorbar(im2, ax=e, shrink=0.9, aspect=18, pad=0.02, ticks=[-1, 0, 1])
    cb2.set_label("Correlation $r$"); cb2.outline.set_visible(False); cb2.ax.tick_params(length=0)

    # (f) overlapping histograms + mean lines
    x1, x2 = rng.normal(0, 1, 800), rng.normal(1.4, 0.8, 800)
    bins = np.linspace(-3.5, 4.5, 36)
    f.hist(x1, bins, color=C["indigo"], alpha=0.65, label="Before")
    f.hist(x2, bins, color=C["clay"], alpha=0.65, label="After")
    for xs, col in ((x1, C["indigo"]), (x2, C["clay"])):
        f.axvline(xs.mean(), color=col, lw=1.0, ls="--")
    f.set(xlabel="Value", ylabel="Count"); f.legend(loc="upper right")

    ps.panel_labels(axs)
    return fig


# --------------------------------------------------------------------------------------
# Figure 2: seaborn, double column, six panels
# --------------------------------------------------------------------------------------
def fig_seaborn():
    n = 150
    df = pd.DataFrame({
        "group": rng.choice(["Ctrl", "Low", "Mid", "High"], n * 2),
        "cond": np.repeat(["Untreated", "Treated"], n),
    })
    shift = df["group"].map({"Ctrl": 0, "Low": 0.4, "Mid": 0.9, "High": 1.4}).to_numpy()
    df["value"] = rng.normal(1, 0.5, n * 2) + shift * np.where(df["cond"] == "Treated", 1.0, 0.25)

    t = np.tile(np.arange(0, 12), 30)
    ts = pd.DataFrame({"t": t, "method": np.repeat(["Ours", "Baseline", "Ablation"], 120)})
    ts["score"] = (ts["method"].map({"Ours": 0.88, "Baseline": 0.76, "Ablation": 0.80}) *
                   (1 - np.exp(-(ts["t"] + 1) / 3.0)) + rng.normal(0, 0.03, len(ts)))

    kd = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 500), rng.normal(2, 0.7, 500), rng.normal(-1.5, 0.5, 500)],
                       "class": np.repeat(["Class 1", "Class 2", "Class 3"], 500)})

    sc = pd.DataFrame({"x": rng.uniform(0, 10, 160)})
    sc["species"] = np.where(sc.x < 3.3, "S1", np.where(sc.x < 6.6, "S2", "S3"))
    sc["y"] = 0.6 * sc.x + rng.normal(0, 1.1, 160) + sc["species"].map({"S1": 2, "S2": 0, "S3": -1})

    fig, axs = plt.subplots(2, 3, figsize=ps.figsize("double", height=102))
    a, b, c, d, e, f = axs.ravel()

    sns.violinplot(df, x="group", y="value", hue="cond", split=True, inner="quart", gap=0.05,
                   palette=ps.palette_hex(2), linewidth=0.6, order=["Ctrl", "Low", "Mid", "High"],
                   ax=a, saturation=1)
    a.set(xlabel="", ylabel="Measurement"); sns.move_legend(a, "upper left", title=None, ncols=2)

    sns.lineplot(ts, x="t", y="score", hue="method", style="method", errorbar="sd", palette=ps.palette_hex(3),
                 hue_order=["Ours", "Baseline", "Ablation"], err_kws=dict(alpha=0.18, lw=0), ax=b,
                 markers=True, dashes=False, markersize=3.5, linewidth=1.2)
    b.set(xlabel="Epoch", ylabel="Score"); sns.move_legend(b, "lower right", title=None)

    sns.kdeplot(kd, x="x", hue="class", fill=True, common_norm=False, palette=ps.palette_hex(3),
                alpha=0.45, linewidth=1.0, ax=c)
    c.set(xlabel="Feature", ylabel="Density"); sns.move_legend(c, "upper right", title=None)

    order = ["S1", "S2", "S3"]
    sns.scatterplot(sc, x="x", y="y", hue="species", style="species", hue_order=order, style_order=order,
                    palette=ps.palette_hex(3), s=14, ax=d)
    for i, (sp, g) in enumerate(sc.groupby("species")):
        sns.regplot(g, x="x", y="y", scatter=False, ci=None, color=ps.CATEGORICAL[i],
                    line_kws=dict(lw=1.0), ax=d)
    d.set(xlabel="Dose", ylabel="Response"); sns.move_legend(d, "upper left", title=None, ncols=3,
                                                              columnspacing=0.6, handletextpad=0.1)

    mat = pd.DataFrame(rng.normal(size=(7, 7)) @ rng.normal(size=(7, 7)) * 0.4,
                       index=list("ABCDEFG"), columns=list("ABCDEFG"))
    mat = (mat - mat.values.mean()) / mat.values.std() * 0.9
    sns.heatmap(mat, cmap=ps.DIV, center=0, vmin=-2, vmax=2, annot=True, fmt=".1f",
                annot_kws=dict(fontsize=5.5), linewidths=0.6, linecolor="white", square=True,
                cbar_kws=dict(shrink=0.8, aspect=16, ticks=[-2, 0, 2], label="Log$_2$ fold change"), ax=e)
    e.tick_params(length=0); e.collections[0].colorbar.outline.set_visible(False)
    e.collections[0].colorbar.ax.tick_params(length=0)

    bars = pd.DataFrame({"model": np.repeat(["Ours", "A", "B", "C"], 25), "acc": np.concatenate(
        [rng.normal(m, 0.02, 25) for m in (0.91, 0.84, 0.82, 0.78)])})
    sns.barplot(bars, x="model", y="acc", color=C["indigo"], errorbar="sd", err_kws=dict(lw=0.8),
                capsize=0.15, width=0.55, ax=f)
    for i, p in enumerate(f.patches):             # ours highlighted, the rest muted in grey
        p.set_facecolor(C["indigo"] if i == 0 else ps.tint(C["grey"], 0.35))
    f.set(xlabel="", ylabel="Accuracy", ylim=(0.6, 1.0))

    ps.panel_labels(axs)
    return fig


# --------------------------------------------------------------------------------------
# Figure 3: single column (89 mm) -- what a typical journal figure looks like at final size
# --------------------------------------------------------------------------------------
def fig_single():
    fig, ax = plt.subplots(figsize=ps.figsize("single", aspect=0.75))
    x = rng.uniform(0, 10, 60)
    y = 0.8 * x + 1 + rng.normal(0, 0.9, 60)
    p = np.polyfit(x, y, 1); xs = np.linspace(0, 10, 50)
    resid = y - np.polyval(p, x); r2 = 1 - resid.var() / y.var()
    ax.fill_between(xs, np.polyval(p, xs) - 1.1, np.polyval(p, xs) + 1.1, color=ps.tint(C["indigo"], 0.78), lw=0)
    ax.scatter(x, y, s=12, color=C["indigo"], label="Samples")
    ax.plot(xs, np.polyval(p, xs), color=C["clay"], label="Linear fit")
    ax.text(0.04, 0.95, rf"$y = {p[0]:.2f}\,x + {p[1]:.2f}$" + "\n" + rf"$R^2 = {r2:.2f}$",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set(xlabel=r"Concentration $c$ ($\mu$M)", ylabel=r"Signal $\Delta F/F_0$ (%)")
    ax.legend(loc="lower right")
    return fig


if __name__ == "__main__":
    ps.use(latex=LATEX)
    suffix = "_latex" if LATEX else ""
    for name, fn in (("fig_palette", fig_palette), ("fig_matplotlib", fig_matplotlib),
                     ("fig_seaborn", fig_seaborn), ("fig_single", fig_single)):
        fig = fn()
        print(ps.save(fig, name + suffix))
        plt.close(fig)
