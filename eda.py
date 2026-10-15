import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.utils import npy_preprocessor
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


os.makedirs(OUT_DIR, exist_ok=True)
def plot_four_way_comparison(y_orig_minmax, y_filt_minmax, y_orig_std, y_filt_std,
                             orig_count, filt_count, save_path="out/four_way_distribution.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    configs = [
        # (row, col, data, title, xlabel, color, vlines)
        (0, 0, y_orig_minmax, f"1. Original Data \n(N = {orig_count:,})", 
         "Scaled Value [0, 1]", "#2b5c8f"),
        (0, 1, y_filt_minmax, f"2. Filtered Data (|Z| <= 3) \n(N = {filt_count:,})", 
         "Scaled Value [0, 1]", "#1f77b4"),
        (1, 0, y_orig_std, f"3. Original Data — Standardized (Z-Score)\n(N = {orig_count:,})", 
         "Standardized Value (Mean=0, Std=1)", "#2ca02c"),
        (1, 1, y_filt_std, f"4. Filtered Data (|Z| <= 3) — Re-Standardized\n(N = {filt_count:,})", 
         "Standardized Value (Mean=0, Std=1)", "#2e8b57"),
    ]

    for row, col, data, title, xlabel, color in configs:
        ax = axes[row, col]
        
        # Calculate summary metrics
        d_min = np.min(data)
        d_max = np.max(data)
        d_mean = np.mean(data)
        d_std = np.std(data)
        d_median = np.median(data)

        # Plot distribution
        sns.histplot(data, bins=120, ax=ax, color=color, edgecolor="black", linewidth=0.2)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Count / Frequency", fontsize=11)

        # Reference lines
        ax.axvline(d_mean, color="orange", linestyle="-", linewidth=1.5, label=f"Mean ({d_mean:.2f})")
        ax.axvline(d_median, color="red", linestyle=":", linewidth=1.8, label=f"Median ({d_median:.2f})")

        # Stats summary box
        info_box = (
            f"Count:  {len(data):,}\n"
            f"Min:    {d_min:.4f}\n"
            f"Max:    {d_max:.4f}\n"
            f"Mean:   {d_mean:.4f}\n"
            f"Std:    {d_std:.4f}\n"
            f"Median: {d_median:.4f}"
        )
        ax.text(
            0.96, 0.94, info_box,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=9.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.92)
        )
        ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[✓] 4-way comparison plot saved successfully to: {save_path}")

def plot_molecule(molecule_id, molecule_inchi, molecule, grid_limit=5.0, save_path=None):
    """
    Plot a molecule as a 3D scatterplot inside a fixed-size cubic grid.

    Parameters:
        molecule_id (str/int): Unique identifier for the molecule.
        molecule_inchi (str): InChI string for display.
        molecule: Array-like of shape (N, 8) with coords and one-hot atom types.
        grid_limit (float): Half-width of the cube. Sets axes to [-grid_limit, +grid_limit].
                            Default 5.0 Å easily accommodates standard QM9 molecules.
        save_path (str, optional): Target output path for the rendered plot.
    """
    if save_path is None:
        save_path = os.path.join(OUT_DIR, f"molecule_{molecule_id}_3d.png")

    molecule = np.asarray(molecule)

    x = molecule[:, 0]
    y = molecule[:, 1]
    z = molecule[:, 2]

    # One-hot encoded atom types
    atom_types = molecule[:, 3:8]
    elements = ["H", "C", "N", "O", "F"]

    # CPK element coloring and sizes
    colors = {
        "H": "#D3D3D3",
        "C": "#333333",
        "N": "#305CDE",
        "O": "#E53935",
        "F": "#2EAD55",
    }
    sizes = {
        "H": 40,
        "C": 100,
        "N": 100,
        "O": 100,
        "F": 100,
    }

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each element
    for i, element in enumerate(elements):
        mask = atom_types[:, i] == 1
        if np.any(mask):
            ax.scatter(
                x[mask],
                y[mask],
                z[mask],
                c=colors[element],
                s=sizes[element],
                label=element,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                depthshade=True,
            )

    ax.set_title(
        f"Molecule ID: {molecule_id}\n3D Molecular Structure\n{molecule_inchi}",
        fontsize=10,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_zlabel("Z Coordinate (Å)")
    ax.legend(title="Elements", loc="upper right")

    # --- FIXED-SIZE GRID ENFORCEMENT ---
    # Set identical fixed limits across all 3 axes (e.g., -5.0 Å to +5.0 Å)
    ax.set_xlim(-grid_limit, grid_limit)
    ax.set_ylim(-grid_limit, grid_limit)
    ax.set_zlim(-grid_limit, grid_limit)

    # Optional: Clean, consistent tick steps across all 3 dimensions
    step = 2.5 if grid_limit <= 5.0 else 5.0
    ticks = np.arange(-grid_limit, grid_limit + 1e-5, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Lock geometry to a perfect 1:1:1 cubic aspect ratio
    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[✓] 3D molecule scatterplot (fixed ±{grid_limit}Å grid) saved to: {save_path}")




def plot_len_histogram(molecules, save_path=None):
    """Computes the number of real atoms in each molecule (based on non-zero atom

    type features in columns 3:8) and plots a histogram of molecule lengths.

    Parameters
    ----------
    molecules : iterable or np.ndarray
        Array/Series of molecule matrices, each typically of shape (27, 8+)
        where cols 0:3 are coords (x, y, z) and cols 3:8 are one-hot atom
        types.
    save_path : str, optional
        Path where the histogram figure should be saved. Displays with
        plt.show() if None.
    """
    lengths = []

    for mol in molecules:
        mol = np.asarray(mol)

        # Slice the one-hot encoded / feature channels for atom types (cols 3 to 7 inclusive)
        atom_types = mol[:, 3:8]

        # An atom exists if any of its one-hot channels are non-zero (or sum > 0)
        # Using any(axis=-1) or sum(axis=-1) detects non-empty atom slots
        atom_mask = np.any(atom_types != 0, axis=-1)
        num_atoms = int(np.sum(atom_mask))

        lengths.append(num_atoms)

    lengths = np.array(lengths)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Use discrete integer bin edges from 0.5 to 27.5 so each integer [1..27] gets a centered bar
    min_len = 1
    max_len = 27
    bins = np.arange(min_len - 0.5, max_len + 1.5, 1)

    counts, edges, patches = ax.hist(
        lengths,
        bins=bins,
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        rwidth=0.85,
    )

    # Styling and annotations
    ax.set_title(
        f"Distribution of Molecule Sizes in QM9 (N = {len(lengths):,})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Number of Atoms", fontsize=11)
    ax.set_ylabel("Molecule Count", fontsize=11)
    ax.set_xlim(min_len - 1, max_len + 1)
    ax.set_xticks(np.arange(min_len, max_len + 1, 2))  # Tick every 2 units
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate summary stats in a text box
    stats_text = (
        f"Min: {lengths.min()}\n"
        f"Max: {lengths.max()}\n"
        f"Mean: {lengths.mean():.2f}\n"
        f"Median: {int(np.median(lengths))}"
    )
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
        ),
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Histogram saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_element_counts(df):
    """Counts the number of H, C, N, O, F atoms in each molecule.

    Assumes df['xyz'] contains an (N_atoms, 8) array where:
        columns 3:8 correspond to [H, C, N, O, F] one-hot flags.
    """
    elements = ["H", "C", "N", "O", "F"]

    # Sum across atoms (axis=0 of each xyz array) for one-hot columns [3:8]
    # Result per molecule is an array of shape (5,) -> [count_H, count_C, count_N, count_O, count_F]
    counts_matrix = np.array([np.sum(mol[:, 3:8], axis=0) for mol in df])

    count_df = pd.DataFrame(counts_matrix, columns=elements, dtype=int)
    return count_df


def plot_element_occurrence_histograms(
    df, save_path="out/element_occurrence_histograms.png"
):
    """Plots 5 subplots (one for each element: H, C, N, O, F) using 27 bins

    representing atom counts from 0 to 26.
    """
    counts_df = compute_element_counts(df)
    elements = ["H", "C", "N", "O", "F"]

    element_colors = {
        "H": "#7F7F7F",  # Gray
        "C": "#2B2B2B",  # Charcoal/Black
        "N": "#1F77B4",  # Blue
        "O": "#D62728",  # Red
        "F": "#2CA02C",  # Green
    }

    # Bins for discrete counts 0 through 26 (27 bins: [0, 1), [1, 2), ..., [26, 27))
    bin_edges = np.arange(28) - 0.5  # Centers bars directly on integer values

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    for i, elem in enumerate(elements):
        ax = axes[i]
        vals = counts_df[elem].values

        total_with_elem = np.sum(vals > 0)
        pct_with_elem = (total_with_elem / len(df)) * 100
        max_atoms = np.max(vals)

        # Plot discrete histogram
        sns.histplot(
            vals,
            bins=bin_edges,
            ax=ax,
            color=element_colors[elem],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )

        ax.set_title(
            f"Element: {elem} — Molecules containing ≥1: {total_with_elem:,} ({pct_with_elem:.1f}%) | Max in single mol: {max_atoms}",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax.set_ylabel("Molecules", fontsize=11)
        ax.set_xlim(-0.5, 26.5)
        ax.set_xticks(range(27))

        # Annotate peak counts on bars that have meaningful frequencies
        counts, _ = np.histogram(vals, bins=bin_edges)
        for x_val, count in enumerate(counts):
            if count > (len(df) * 0.01):  # Label bars holding >1% of data
                ax.text(
                    x_val,
                    count + (max(counts) * 0.02),
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=45,
                )

    axes[-1].set_xlabel("Number of Atoms per Molecule", fontsize=12)

    plt.suptitle(
        f"QM9 Dataset: Element Atom Count Distribution per Molecule (N = {len(df):,})",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Element histogram plot saved to: {save_path}")



def main():
    policy_path = sys.argv[1] 
    optimal_config_values = json.load(open(policy_path))

    df_full = npy_preprocessor("qm9_filtered.npy")
    print(df_full.head(5))

    print(df_full.iloc[0]['inchi'])
    xyz_arr = df_full.iloc[0]['xyz']

    # Define the 8 QM9 columns
    cols = ['X', 'Y', 'Z', 'H', 'C', 'N', 'O', 'F']

    # Display first 10 rows as a formatted table
    print(pd.DataFrame(xyz_arr[:10], columns=cols))


    dup_mask = df_full.duplicated(subset=["inchi"], keep=False)
    duplicates = df_full[dup_mask]

    dup_counts = df_full.groupby("inchi").size()

    # Filter for InChIs with strictly more than 3 conformations (>= 4)
    inchis_gt_3 = dup_counts[dup_counts > 3].index
    print(f"Total duplicate rows found: {len(duplicates)}")

    if not duplicates.empty:

        sample_inchi = inchis_gt_3[0]
        sample_dups = duplicates[duplicates["inchi"] == sample_inchi].reset_index(drop=True)

        print(f"Found {len(sample_dups)} conformations for sample InChI: {sample_inchi}")

        for idx in range(len(sample_dups)):
            row = sample_dups.iloc[idx]
            mol_id = row["index"]
            mol_inchi = row["inchi"]
            mol_xyz = row["xyz"]
            save_dest = os.path.join(OUT_DIR, f"duplicate_conformation_{mol_id}.png")
            plot_molecule(molecule_id=mol_id, molecule_inchi=mol_inchi, molecule=mol_xyz, save_path=save_dest)

    # 3. Deduplicate dataset retaining the first instance
    df_deduped = df_full.drop_duplicates(subset=["inchi"], keep="first").reset_index(drop=True)
    print(f"Dataset deduplicated: {len(df_full):,} -> {len(df_deduped):,} entries.")
    df_full = df_full.drop_duplicates(subset=["inchi"], keep="first").reset_index(drop=True)

    len_save_dest = os.path.join(OUT_DIR, f"molecule_size_distribution.png")
    element_appearance_dest = os.path.join(OUT_DIR, f"element_appearance_distribution.png")

    plot_len_histogram(df_full['xyz'], save_path=len_save_dest)
    plot_element_occurrence_histograms(df_full['xyz'], save_path=element_appearance_dest)

    
    # 1. Extract raw 589nm target values
    y_raw = np.stack(df_full["rotation"].values)[:, 1].astype(float)
    orig_count = len(y_raw)

    # 2. Vectorized initial Mean, Std, and Z-score calculation
    mean_orig = np.mean(y_raw)
    std_orig = np.std(y_raw)
    z_scores_orig = (y_raw - mean_orig) / std_orig

    # 3. Identify and remove outliers (|Z| > 3.0)
    outlier_mask = np.abs(z_scores_orig) > optimal_config_values['train_omit_z_score']
    outlier_count = np.sum(outlier_mask)
    y_filt = y_raw[~outlier_mask]
    filt_count = len(y_filt)

    print(f"--- Population Filtering Summary ---")
    print(f"Original Count  : {orig_count:,}")
    print(f"Removed (|Z|>3) : {outlier_count:,} ({(outlier_count / orig_count) * 100:.2f}%)")
    print(f"Retained Count  : {filt_count:,}\n")


    y_orig_minmax = y_raw.reshape(-1, 1).flatten()

    y_filt_minmax = y_filt.reshape(-1, 1).flatten()

    # 4C. Original Standardized
    std_scaler_orig = StandardScaler()
    y_orig_std = std_scaler_orig.fit_transform(y_raw.reshape(-1, 1)).flatten()

    # 4D. Filtered Standardized (re-standardized on clean cohort)
    std_scaler_filt = StandardScaler()
    y_filt_std = std_scaler_filt.fit_transform(y_filt.reshape(-1, 1)).flatten()

    # 5. Render and save the 2x2 comparison figure
    out_image_path = os.path.join(OUT_DIR, "four_way_distribution.png")
    plot_four_way_comparison(
        y_orig_minmax, 
        y_filt_minmax, 
        y_orig_std, 
        y_filt_std, 
        orig_count, 
        filt_count, 
        save_path=out_image_path
    )


if __name__ == "__main__":
    main()