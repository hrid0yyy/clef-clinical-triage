import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Measured outputs for the illustrative psoriasis narrative
# (gpu-server/xai/xai_perturbation_results.json, case 1).
NARRATIVE = ("I have an itchy red rash on my arms with dry silvery scales "
             "peeling off. My skin is cracked and bleeds occasionally.")
CLAUSES = [
    ("$s_1$", "itchy red rash", 0.0000119),
    ("$s_2$", "dry silvery scales peeling off", 99.8823),
    ("$s_3$", "skin is cracked", 0.00025),
    ("$s_4$", "bleeds occasionally", -0.0000119),
]
TOP_CLASSES = [
    ("Psoriasis", 0.9999986),
    ("Arthritis", 1.255e-7),
    ("Drug reaction", 1.179e-7),
]
FLIP_LABEL, FLIP_PROB = "Impetigo", 0.9959

C_EDGE = "#333333"
C_TEXT = "#222222"
C_MUTED = "#7F7F7F"
C_FROZEN = "#E6E6E6"
C_TRAIN = "#C6DBEF"
C_ACCENT = "#C44E52"
C_BAR = "#4C72B0"
C_HL = "#F6D5D6"


def box(ax, x, y, w, h, text, fc="white", fs=6.6, ec=C_EDGE, lw=0.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, transform=ax.transAxes, weight=weight, linespacing=1.25)


def down_arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=6,
                                 lw=0.6, color=C_EDGE, transform=ax.transAxes))


def panel_label(ax, label, title):
    ax.text(0.0, 1.05, f"({label}) {title}", transform=ax.transAxes, fontsize=7.2,
            ha="left", va="bottom")


def create_overview_figure(save_dirs=['figures']):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.4,
    })

    fig = plt.figure(figsize=(7.4, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 0.95, 1.0], wspace=0.7,
                          left=0.015, right=0.985, top=0.86, bottom=0.17)

    # (a) Input narrative and clause segmentation
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    panel_label(ax, "a", "Input narrative $X$")
    ax.add_patch(Rectangle((0, 0.56), 1, 0.38, fc="white", ec=C_EDGE, lw=0.6,
                           transform=ax.transAxes))
    ax.text(0.04, 0.75,
            "“I have an itchy red rash on my\narms with dry silvery scales\n"
            "peeling off. My skin is cracked\nand bleeds occasionally.”",
            transform=ax.transAxes, fontsize=6.2, style="italic", va="center", linespacing=1.35)
    ax.text(0.0, 0.48, "Clause segmentation $S=\\{s_1,\\dots,s_4\\}$", transform=ax.transAxes,
            fontsize=6.2, color=C_MUTED, va="center")
    for i, (sym, clause, _) in enumerate(CLAUSES):
        y = 0.36 - i * 0.095
        hl = i == 1
        ax.add_patch(Rectangle((0, y - 0.04), 1, 0.08, fc=C_HL if hl else "white",
                               ec=C_ACCENT if hl else "#BBBBBB", lw=0.5, transform=ax.transAxes))
        ax.text(0.03, y, sym, transform=ax.transAxes, fontsize=6.4, va="center")
        ax.text(0.15, y, clause, transform=ax.transAxes, fontsize=6.2, va="center")

    # (b) Model
    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    panel_label(ax, "b", "Decision model (Clef-Flash)")
    box(ax, 0.06, 0.82, 0.88, 0.11, "Schema: state $X$, $K{=}24$ options", fs=5.9)
    down_arrow(ax, 0.5, 0.82, 0.755)
    ax.add_patch(Rectangle((0.0, 0.47), 1.0, 0.28, fc=C_FROZEN, ec=C_EDGE, lw=0.6,
                           transform=ax.transAxes))
    ax.text(0.5, 0.685, "Qwen3.5-9B backbone (4-bit, frozen)", ha="center", va="center",
            fontsize=5.9, transform=ax.transAxes)
    box(ax, 0.06, 0.50, 0.88, 0.11, "LoRA: $W_q, W_k, W_v, W_o$; $r{=}16$", fc=C_TRAIN, fs=5.8)
    down_arrow(ax, 0.5, 0.47, 0.405)
    box(ax, 0.06, 0.295, 0.88, 0.11, "Per-token states $H$ + option spans", fs=5.9)
    down_arrow(ax, 0.5, 0.295, 0.23)
    box(ax, 0.06, 0.06, 0.88, 0.17,
        "JointSchemaHead\nroute + decode; $z_k$; softmax", fc=C_TRAIN, fs=5.8)
    ax.add_patch(Rectangle((0.06, -0.075), 0.05, 0.04, fc=C_FROZEN, ec=C_EDGE, lw=0.5,
                           transform=ax.transAxes, clip_on=False))
    ax.text(0.13, -0.055, "frozen", fontsize=6, va="center", transform=ax.transAxes)
    ax.add_patch(Rectangle((0.40, -0.075), 0.05, 0.04, fc=C_TRAIN, ec=C_EDGE, lw=0.5,
                           transform=ax.transAxes, clip_on=False))
    ax.text(0.47, -0.055, "trainable", fontsize=6, va="center", transform=ax.transAxes)
    ax.text(0.5, -0.16, "one forward pass, no decoding", fontsize=6,
            color=C_MUTED, ha="center", va="center", transform=ax.transAxes)

    # (c) Predicted distribution
    ax = fig.add_subplot(gs[2])
    panel_label(ax, "c", "Prediction $P(c\\mid X)$")
    names = [n for n, _ in TOP_CLASSES][::-1]
    probs = [p for _, p in TOP_CLASSES][::-1]
    colors = [C_MUTED, C_MUTED, C_BAR]
    ax.barh(range(3), probs, height=0.55, color=colors, left=1e-9)
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 3)
    ax.set_xticks([1e-8, 1e-4, 1])
    ax.set_xticklabels(["$10^{-8}$", "$10^{-4}$", "$1$"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(names)
    ax.set_ylim(-1.1, 2.6)
    ax.set_xlabel("probability (log scale)", fontsize=6.4, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(2e-9, 2, "$>0.9999$", fontsize=6, color="white", va="center", ha="left")
    ax.text(2e-9, -0.7, "other 21 classes $<10^{-7}$", fontsize=5.8, color=C_MUTED,
            ha="left", va="center")

    # (d) LOSO clause ablation
    ax = fig.add_subplot(gs[3])
    panel_label(ax, "d", "Clause ablation (LOSO)")
    drops = [d for _, _, d in CLAUSES][::-1]
    syms = [s for s, _, _ in CLAUSES][::-1]
    cols = [C_ACCENT if d > 1 else C_MUTED for d in drops]
    ax.barh(range(4), [max(d, 0) for d in drops], height=0.55, color=cols)
    ax.set_yticks(range(4))
    ax.set_yticklabels(syms)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(-0.6, 4.3)
    ax.set_xlabel("$\\Delta P$ (percentage points)", fontsize=6.4, labelpad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for yi, d in enumerate(drops):
        if d > 1:
            ax.text(d - 2, yi, f"{d:.2f}", color="white", fontsize=6, ha="right", va="center")
        else:
            ax.text(2, yi, "$|\\Delta P|<0.001$", color=C_MUTED, fontsize=5.8, va="center")
    ax.text(2, 3.85, f"without $s_2$: {FLIP_LABEL} ({FLIP_PROB:.3f})",
            fontsize=5.8, color=C_ACCENT, ha="left", va="center")

    # Arrows between panels (figure coordinates)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ax_a, ax_b, ax_c, _ = fig.axes
    for left, right in [(ax_a, ax_b), (ax_b, ax_c)]:
        x0 = inv.transform(left.get_tightbbox().p1)[0] + 0.006
        x1 = inv.transform(right.get_tightbbox().p0)[0] - 0.006
        fig.add_artist(FancyArrowPatch((x0, 0.55), (x1, 0.55), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, lw=0.7, color=C_EDGE))

    for sdir in save_dirs:
        abs_sdir = os.path.abspath(sdir)
        os.makedirs(abs_sdir, exist_ok=True)
        pdf_path = os.path.join(abs_sdir, 'figure_overview.pdf')
        png_path = os.path.join(abs_sdir, 'figure_overview.png')
        plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
        print(f"Saved: {pdf_path} and {png_path}")

    plt.close()


if __name__ == '__main__':
    create_overview_figure()
