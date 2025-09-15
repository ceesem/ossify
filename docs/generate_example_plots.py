#!/usr/bin/env python3
"""
Generate example plots for ossify documentation using real data.
"""

import matplotlib.pyplot as plt
import numpy as np
import ossify as osy
from ossify.algorithms import strahler_number
from ossify.plot import (
    plot_morphology_2d,
    plot_cell_multiview,
    single_panel_figure,
    add_scale_bar,
)

# Suppress matplotlib warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Loading example cell...")

    # Load a cell from the provided example
    cell = osy.load_cell(
        "https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy"
    )

    print(f"Loaded cell with {cell.skeleton.n_vertices} skeleton vertices")
    print(f"Cable length: {cell.skeleton.cable_length():.0f} nm")
    print(f"Available skeleton features: {cell.skeleton.feature_names}")
    print(f"Available annotations: {cell.annotations.names}")

    # Add some analysis
    strahler_vals = strahler_number(cell)
    cell.skeleton.add_feature(strahler_vals, "strahler_number")

    # Map volume data
    if "size_nm3" in cell.graph.feature_names:
        volume = cell.graph.map_features_to_layer(
            "size_nm3", layer="skeleton", agg="sum"
        )
        cell.skeleton.add_feature(volume, "volume")

    # 1. Basic morphology plot
    print("\nGenerating basic morphology plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_morphology_2d(
        cell,
        projection="xy",
        color="compartment",
        palette={1: "navy", 2: "tomato", 3: "black"},
        linewidth="radius",
        linewidth_norm=(100, 500),
        widths=(0.5, 5),
        root_marker=True,
        ax=ax,
    )
    ax.set_title("Neuron Morphology - Compartment Classification", fontsize=14)
    ax.set_xfeature("X (nm)")
    ax.set_yfeature("Y (nm)")
    plt.tight_layout()
    plt.savefig("docs/images/basic_morphology.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Strahler number visualization
    print("Generating Strahler analysis plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_morphology_2d(
        cell,
        projection="xy",
        color="strahler_number",
        palette="viridis",
        linewidth=2,
        root_marker=True,
        root_color="red",
        ax=ax,
    )
    ax.set_title("Strahler Order Analysis", fontsize=14)
    ax.set_xfeature("X (nm)")
    ax.set_yfeature("Y (nm)")
    plt.tight_layout()
    plt.savefig("docs/images/strahler_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Multi-view plot
    print("Generating multi-view plot...")
    axes = plot_cell_multiview(
        cell,
        layout="three_panel",
        color="compartment",
        palette={1: "navy", 2: "tomato", 3: "black"},
        linewidth="radius",
        linewidth_norm=(100, 500),
        widths=(0.5, 3),
        units_per_inch=100_000,
        despine=True,
    )

    # Add titles to each view
    view_titles = {
        "xy": "Front View (XY)",
        "xz": "Side View (XZ)",
        "zy": "Side View (ZY)",
    }
    for proj, ax in axes.items():
        ax.set_title(view_titles[proj], fontsize=12)

    plt.savefig("docs/images/multiview_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Radius visualization
    print("Generating radius visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_morphology_2d(
        cell,
        projection="xy",
        color="radius",
        palette="plasma",
        linewidth="radius",
        linewidth_norm=(100, 500),
        widths=(0.5, 8),
        ax=ax,
    )
    ax.set_title("Radius Variation", fontsize=14)
    ax.set_xfeature("X (nm)")
    ax.set_yfeature("Y (nm)")
    plt.tight_layout()
    plt.savefig("docs/images/radius_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Masked visualization
    print("Generating masked visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original
    plot_morphology_2d(
        cell,
        projection="xy",
        color="compartment",
        palette={1: "lightblue", 2: "lightcoral", 3: "lightgray"},
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set_title("Complete Morphology", fontsize=14)

    # Dendrite only (compartment == 3)
    with cell.mask_context(
        "skeleton", cell.skeleton.features["compartment"] == 3
    ) as masked_cell:
        plot_morphology_2d(
            masked_cell, projection="xy", color="black", linewidth=3, ax=axes[1]
        )
    axes[1].set_title("Dendrite Only (Masked)", fontsize=14)

    for ax in axes:
        ax.set_xfeature("X (nm)")
        ax.set_yfeature("Y (nm)")

    plt.tight_layout()
    plt.savefig("docs/images/masking_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Publication-quality figure with scale bar
    print("Generating publication-quality figure...")

    # Convert to micrometers for better scale
    display_cell = cell.transform(lambda x: x / 1000)
    display_cell.name = f"{cell.name}_display"

    fig, ax = single_panel_figure(
        data_bounds_min=display_cell.skeleton.bbox[0],
        data_bounds_max=display_cell.skeleton.bbox[1],
        units_per_inch=50,  # 50 μm per inch
        despine=True,
        dpi=300,
    )

    plot_morphology_2d(
        display_cell,
        projection="xy",
        color="compartment",
        palette={1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"},
        linewidth="radius",
        linewidth_norm=(0.1, 0.5),  # Adjusted for μm scale
        widths=(0.5, 4),
        root_marker=True,
        root_color="red",
        ax=ax,
    )

    # Add scale bar
    add_scale_bar(
        ax=ax,
        length=50,  # 50 μm
        position=(0.05, 0.05),
        feature="50 μm",
        color="black",
        linewidth=3,
        fontsize=12,
    )

    plt.savefig("docs/images/publication_figure.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nAll plots generated successfully!")
    print("Generated files:")
    print("  - docs/images/basic_morphology.png")
    print("  - docs/images/strahler_analysis.png")
    print("  - docs/images/multiview_plot.png")
    print("  - docs/images/radius_visualization.png")
    print("  - docs/images/masking_example.png")
    print("  - docs/images/publication_figure.png")


if __name__ == "__main__":
    main()
