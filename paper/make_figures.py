"""Generate the manuscript figures from the DiffBio Gate-2 benchmark results.

All numbers are the summary statistics (mean +/- std over 3 seeds unless noted) of the
runs described in the paper; each block cites its source run. Figures are written as
vector PDFs into ``paper/figures/`` for inclusion in the LaTeX manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FROZEN = "#4C72B0"
JOINT = "#C44E52"
ACCENT = "#55A868"

# --- Main Gate-2 result: macro-F1 and rare-type F1, frozen vs joint --------------
# Source runs: gate2_joint_pipeline.json (TS120k dispersion); gate2_ts120k_supervised,
# gate2_tsall_dispersion, gate2_tsall_supervised (gate2_strengthen.py).
CONDITIONS = [
    # label, frozen_macro, frozen_std, joint_macro, joint_std, frozen_rare, joint_rare
    ("TS-120k\n59 types\ndispersion", 0.7304, 0.0054, 0.7826, 0.0034, 0.563, 0.625),
    ("TS-120k\n59 types\nsupervised", 0.7398, 0.0134, 0.7989, 0.0000, 0.595, 0.648),
    ("TS-193k\n87 types\ndispersion", 0.6497, 0.0060, 0.7518, 0.0000, 0.403, 0.534),
]


def fig_main_result() -> None:
    """Grouped bars: frozen vs joint, overall macro-F1 and rare-type F1."""
    labels = [c[0] for c in CONDITIONS]
    x = np.arange(len(labels))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    frozen = [c[1] for c in CONDITIONS]
    frozen_err = [c[2] for c in CONDITIONS]
    joint = [c[3] for c in CONDITIONS]
    joint_err = [c[4] for c in CONDITIONS]
    axes[0].bar(
        x - width / 2, frozen, width, yerr=frozen_err, capsize=3, label="Frozen PCA", color=FROZEN
    )
    axes[0].bar(
        x + width / 2,
        joint,
        width,
        yerr=joint_err,
        capsize=3,
        label="Joint projection",
        color=JOINT,
    )
    for xi, (f, j) in enumerate(zip(frozen, joint, strict=True)):
        axes[0].annotate(
            f"+{100 * (j - f):.1f}", (xi, max(f, j) + 0.02), ha="center", fontsize=8, color=JOINT
        )
    axes[0].set_ylabel("Held-out macro-F1")
    axes[0].set_title("(a) Cell-type annotation accuracy")
    axes[0].set_ylim(0.55, 0.87)

    frozen_rare = [c[5] for c in CONDITIONS]
    joint_rare = [c[6] for c in CONDITIONS]
    axes[1].bar(x - width / 2, frozen_rare, width, label="Frozen PCA", color=FROZEN)
    axes[1].bar(x + width / 2, joint_rare, width, label="Joint projection", color=JOINT)
    for xi, (f, j) in enumerate(zip(frozen_rare, joint_rare, strict=True)):
        axes[1].annotate(
            f"+{100 * (j - f):.1f}", (xi, max(f, j) + 0.015), ha="center", fontsize=8, color=JOINT
        )
    axes[1].set_ylabel("Rare cell-type F1")
    axes[1].set_title("(b) Rare cell types")
    axes[1].set_ylim(0.35, 0.72)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_main_result.pdf")
    plt.close(fig)


def fig_gain_vs_k() -> None:
    """The central regime figure: frozen and joint macro-F1 as a function of k."""
    # Matched frozen/joint macro-F1 across k (TS-120k, dispersion HVG, 3 seeds).
    k_values = [5, 10, 20, 50]
    frozen = {5: (0.5645, 0.0108), 10: (0.7304, 0.0054), 20: (0.7941, 0.0070), 50: (0.8243, 0.0111)}
    joint = {5: (0.7488, 0.0077), 10: (0.7826, 0.0034), 20: (0.8211, 0.0064), 50: (0.8406, 0.0035)}
    soft = (0.8304, 40.3)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ks = np.array(k_values)
    fm = np.array([frozen[k][0] for k in k_values])
    fs = np.array([frozen[k][1] for k in k_values])
    ax.plot(ks, fm, "-o", color=FROZEN, label="Frozen PCA")
    ax.fill_between(ks, fm - fs, fm + fs, color=FROZEN, alpha=0.15)

    jk = sorted(joint)
    jm = np.array([joint[k][0] for k in jk])
    js = np.array([joint[k][1] for k in jk])
    ax.plot(jk, jm, "-s", color=JOINT, label="Joint projection")
    ax.fill_between(jk, jm - js, jm + js, color=JOINT, alpha=0.15)

    ax.plot(
        [soft[1]],
        [soft[0]],
        "*",
        color=ACCENT,
        markersize=14,
        label=f"Soft-k (learned dim {soft[1]:.0f})",
    )
    for k in jk:
        gain = 100 * (joint[k][0] - frozen[k][0])
        ax.annotate(
            f"+{gain:.1f}pp", (k, joint[k][0] + 0.006), ha="center", fontsize=8, color=JOINT
        )
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("Number of components  $k$")
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Joint gain concentrates at aggressive reduction")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gain_vs_k.pdf")
    plt.close(fig)


def fig_scaling_and_ablation() -> None:
    """Gain vs dataset size (crossover) and the knob-attribution ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # (a) gain (pp) vs training-set size, two modalities. Sources: scRNA joint-
    # preprocessing scaling curve; scATAC gate_atac_deepen.json scaling section.
    scrna_n = np.array([2000, 10000, 33000, 120000, 193000])
    scrna_gain = np.array([-8.6, -2.7, -0.7, 5.2, 10.2])
    scatac_n = np.array([5000, 12000, 30000, 60000, 107672])
    scatac_gain = np.array([-7.1, -2.7, 4.3, 7.5, 12.2])
    dna_n = np.array([8000, 20000, 50000, 100000, 139804])
    dna_gain = np.array([-0.8, -1.7, 1.6, 6.2, 7.5])
    axes[0].axhline(0.0, color="grey", lw=0.8, ls="--")
    axes[0].plot(scrna_n, scrna_gain, "-o", color=JOINT, label="scRNA (PCA)")
    axes[0].plot(scatac_n, scatac_gain, "-s", color=ACCENT, label="scATAC (LSI)")
    axes[0].plot(dna_n, dna_gain, "-^", color="#8172B3", label="DNA (PCA)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training-set size (samples)")
    axes[0].set_ylabel("Joint $-$ frozen macro-F1 (pp)")
    axes[0].set_title("(a) Gain grows with scale (crossover)")
    axes[0].annotate("frozen wins\n(small data)", (3000, -6.5), fontsize=7.5, color=FROZEN)
    axes[0].annotate("joint wins\n(at scale)", (110000, 3.0), fontsize=7.5, color=JOINT)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    # (b) ablation: which learnable knob carries the gain (TS-120k, k=10, 2 seeds).
    variants = ["frozen", "norm", "hvg", "proj", "all"]
    scores = [0.736, 0.734, 0.735, 0.765, 0.765]
    colors = [FROZEN, FROZEN, FROZEN, JOINT, JOINT]
    axes[1].bar(variants, scores, color=colors)
    axes[1].set_ylim(0.72, 0.775)
    axes[1].axhline(scores[0], color="grey", lw=0.8, ls="--")
    axes[1].set_ylabel("Held-out macro-F1")
    axes[1].set_title("(b) The projection carries the gain")
    axes[1].annotate("learnable\nprojection", (3, 0.767), ha="center", fontsize=7.5, color=JOINT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scaling_ablation.pdf")
    plt.close(fig)


def fig_cross_modality() -> None:
    """The joint gain at aggressive reduction across pipelines, modalities and machineries."""
    # (label, gain_pp @ k=5 unless noted, machinery). Sources: gate_sequence/gate_vcc/
    # gate_atac/gate_variant_pool JSON summaries (3 seeds each) and the scRNA gain-vs-k run.
    entries = [
        ("scRNA\nannotation", 18.4, "linear"),
        ("DNA regulatory\nelements", 7.5, "linear"),
        ("Perturb-seq\nidentity", 8.3, "linear"),
        ("scATAC\nannotation", 12.2, "linear"),
        ("Variant calling\n(read pooling)", 16.9, "pool"),
        ("Variant calling\n(no bottleneck)", -1.1, "null"),
    ]
    color = {"linear": JOINT, "pool": ACCENT, "null": "#9AA0A6"}

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    positions = np.arange(len(entries))
    gains = [entry[1] for entry in entries]
    colors = [color[entry[2]] for entry in entries]
    ax.bar(positions, gains, color=colors, width=0.66)
    ax.axhline(0.0, color="grey", lw=0.8)
    for position, gain in zip(positions, gains, strict=True):
        offset = 0.5 if gain >= 0 else -0.9
        ax.annotate(f"{gain:+.1f}", (float(position), gain + offset), ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([entry[0] for entry in entries], fontsize=7.5)
    ax.set_ylabel("Joint $-$ frozen macro-F1 (pp)")
    ax.set_title("The joint gain generalizes across modalities and machineries")
    handles = [
        mpl.patches.Patch(color=JOINT, label="Linear projection"),
        mpl.patches.Patch(color=ACCENT, label="Set pooling (reads)"),
        mpl.patches.Patch(color="#9AA0A6", label="No reduction bottleneck"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    ax.set_ylim(-4, 25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_modality.pdf")
    plt.close(fig)


def main() -> None:
    """Generate all manuscript figures."""
    fig_main_result()
    fig_gain_vs_k()
    fig_scaling_and_ablation()
    fig_cross_modality()
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
