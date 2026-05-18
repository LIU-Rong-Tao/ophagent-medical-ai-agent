from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path("docs/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_workflow_overview():
    steps = [
        "APTOS2019\nDataset",
        "Training\nEngine",
        "Backbone\nBenchmark",
        "Evaluation\nSchema",
        "Grad-CAM\nExplainability",
        "Representation\nComparison",
        "Grounded\nReasoning",
    ]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")

    x_positions = list(range(len(steps)))
    y = 0.5

    for i, (x, label) in enumerate(zip(x_positions, steps)):
        box = FancyBboxPatch(
            (x, y),
            0.82,
            0.34,
            boxstyle="round,pad=0.04,rounding_size=0.05",
            linewidth=1.5,
            facecolor="#F8FAFC",
            edgecolor="#334155",
        )
        ax.add_patch(box)
        ax.text(
            x + 0.41,
            y + 0.17,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (x + 0.86, y + 0.17),
                (x + 0.98, y + 0.17),
                arrowstyle="->",
                mutation_scale=13,
                linewidth=1.5,
                color="#334155",
            )
            ax.add_patch(arrow)

    ax.set_xlim(-0.15, len(steps) - 0.05)
    ax.set_ylim(0.35, 1.0)

    plt.title(
        "OphAgent Workflow: From Benchmark Infrastructure to Grounded Ophthalmology AI",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "workflow_overview.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_foundation_benchmark_overview():
    labels = [
        "ConvNeXt\n10ep",
        "Swin\n10ep",
        "ViT-B\n10ep",
        "ViT-B\nofficial",
        "RETFound\nofficial",
    ]

    macro_f1 = [0.6496, 0.6567, 0.5500, 0.5800, 0.6095]

    colors = [
        "#CBD5E1",
        "#CBD5E1",
        "#CBD5E1",
        "#93C5FD",
        "#2563EB",
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    bars = ax.bar(labels, macro_f1, color=colors, edgecolor="#1E293B", linewidth=1)

    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_ylim(0.50, 0.69)
    ax.set_title(
        "Foundation Representation Benchmark on APTOS2019",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.text(
        0.5,
        0.675,
        "Unified lightweight baseline",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    ax.text(
        3.5,
        0.675,
        "Foundation-style controlled benchmark",
        ha="center",
        fontsize=10,
        color="#1D4ED8",
    )

    ax.axvline(2.5, color="#94A3B8", linestyle="--", linewidth=1)

    for bar, value in zip(bars, macro_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.annotate(
        "+0.0295 Macro F1",
        xy=(4, 0.6095),
        xytext=(3.25, 0.645),
        arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#1D4ED8"),
        fontsize=10,
        fontweight="bold",
        color="#1D4ED8",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "foundation_benchmark_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    save_workflow_overview()
    save_foundation_benchmark_overview()
    print("Saved figures to docs/assets/")