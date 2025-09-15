#!/usr/bin/env python3
"""
Generate the advanced masking visualization example.
"""

import matplotlib.pyplot as plt
import numpy as np
import ossify as osy
from ossify.algorithms import strahler_number
from ossify.plot import plot_cell_2d

# Suppress matplotlib warnings
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Loading example cell...")

    # Load the example cell
    cell = osy.load_cell(
        "https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy"
    )

    # Add Strahler analysis
    strahler_values = strahler_number(cell.skeleton)
    cell.skeleton.add_feature(strahler_values, "strahler_number")

    print("Generating advanced masking visualization...")

    # Mask to axon compartment (compartment == 2) and visualize
    with cell.skeleton.mask_context(
        cell.skeleton.features["compartment"] == 2
    ) as masked_cell:
        fig = plot_cell_2d(
            masked_cell,
            color="strahler_number",  # Color by branching complexity
            palette="coolwarm",  # Blue to red colormap
            linewidth="radius",  # Width varies with radius
            widths=(1, 5.0),  # Min/max line widths
            units_per_inch=100_000,  # Precise scaling
            root_marker=True,  # Show root location
            root_color="k",  # Black root marker
            root_size=50,  # Root marker size
            dpi=100,  # Figure resolution
            projection="xy",  # XY projection
        )

        plt.title("Axon Compartment - Strahler Order Analysis", fontsize=14, pad=20)
        plt.savefig(
            "docs/images/advanced_masking_axon.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Print analysis of masked data
        print(f"Axon cable length: {masked_cell.skeleton.cable_length():.0f} nm")
        print(f"Axon branch points: {len(masked_cell.skeleton.branch_points)}")
        strahler_vals = masked_cell.skeleton.get_feature("strahler_number")
        print(f"Axon Strahler range: {strahler_vals.min()}-{strahler_vals.max()}")

    # Also create dendrite visualization for comparison
    with cell.skeleton.mask_context(
        cell.skeleton.features["compartment"] == 3
    ) as dendrite_cell:
        fig = plot_cell_2d(
            dendrite_cell,
            color="strahler_number",  # Color by branching complexity
            palette="coolwarm",  # Blue to red colormap
            linewidth="radius",  # Width varies with radius
            widths=(1, 5.0),  # Min/max line widths
            units_per_inch=100_000,  # Precise scaling
            root_marker=True,  # Show root location
            root_color="k",  # Black root marker
            root_size=50,  # Root marker size
            dpi=100,  # Figure resolution
            projection="xy",  # XY projection
        )

        plt.title("Dendrite Compartment - Strahler Order Analysis", fontsize=14, pad=20)
        plt.savefig(
            "docs/images/advanced_masking_dendrite.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"Dendrite cable length: {dendrite_cell.skeleton.cable_length():.0f} nm")
        print(f"Dendrite branch points: {len(dendrite_cell.skeleton.branch_points)}")

    print("\nAdvanced masking plots generated successfully!")
    print("Generated files:")
    print("  - docs/images/advanced_masking_axon.png")
    print("  - docs/images/advanced_masking_dendrite.png")


if __name__ == "__main__":
    main()
