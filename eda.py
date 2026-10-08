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

OUT_DIR = os.path.join(PROJECT_ROOT, "out")
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


def main():
    policy_path = sys.argv[1] 
    optimal_config_values = json.load(open(policy_path))

    df_full = npy_preprocessor("qm9_filtered.npy")
    df_full = df_full.drop_duplicates(subset=["inchi"], keep="first").reset_index(drop=True)

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