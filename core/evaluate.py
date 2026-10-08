import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from core.utils import npy_preprocessor, scale_x_coordinates
OUT_DIR = os.path.join(PROJECT_ROOT, "out")



def print_stats(series, name):
    print(f"\n--- Statistics for {name} ---")
    print(f"  Count:  {series.count()}")
    print(f"  Mean:   {series.mean():.2f}")
    print(f"  Std:    {series.std():.2f}")
    print(f"  Min:    {series.min():.2f}")
    print(f"  25% (Q1): {series.quantile(0.25):.2f}")
    print(f"  Median: {series.median():.2f}  <-- (The 'center')")
    print(f"  75% (Q3): {series.quantile(0.75):.2f}")
    print(f"  Max:    {series.max():.2f}")
    
    iqr = series.quantile(0.75) - series.quantile(0.25)
    print(f"  IQR:    {iqr:.2f}")

    clip_min = series.quantile(0.25) - (1.5 * iqr)
    clip_max = series.quantile(0.75) + (1.5 * iqr)
    outliers = (series < clip_min) | (series > clip_max)
    print(f"  Outliers (1.5*IQR): {outliers.sum()} ({outliers.sum() / series.count() * 100:.2f}%)")


def analyze_prediction(filepath, threshold=0.5):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"\n--- ERROR: Prediction file not found at {filepath} ---")
        return None # Return None to signal failure

    true_unscaled = df['true_value_unscaled']
    pred_unscaled = df['prediction_unscaled']
    
    # --- 1. Calculate Standard REGRESSION Metrics ---
    mae = mean_absolute_error(true_unscaled, pred_unscaled)
    mse = mean_squared_error(true_unscaled, pred_unscaled)
    rmse = np.sqrt(mse)
    
    # --- 2. Perform Statistical Test (Paired T-test) ---
    t_stat, p_value = stats.ttest_rel(true_unscaled, pred_unscaled)
    
    # --- 3. HETEROSCEDASTICITY ANALYSIS ---
    errors = true_unscaled - pred_unscaled
    abs_errors = np.abs(errors)
    squared_errors = errors**2
    
    spearman_corr, spearman_p = stats.spearmanr(pred_unscaled, abs_errors)
    
    exog = sm.add_constant(pred_unscaled) # Add an intercept
    bp_test = sm.stats.het_breuschpagan(squared_errors, exog)
    bp_lm_stat, bp_p_value = bp_test[0], bp_test[1]

    # --- 4. CLASSIFICATION & TOLERANCE METRICS ---
    y_true_class = (true_unscaled > 0).astype(int) # 1=Pos, 0=Neg
    y_pred_class = (pred_unscaled > 0).astype(int) # 1=Pos, 0=Neg
    
    sign_accuracy = accuracy_score(y_true_class, y_pred_class)
    sign_f1 = f1_score(y_true_class, y_pred_class, average='binary')
    
    cm = confusion_matrix(y_true_class, y_pred_class)

    dead_zone_count = (np.abs(true_unscaled) <= threshold).sum()
    dead_zone_percent = dead_zone_count / len(true_unscaled)
    tolerance_accuracy = (abs_errors <= threshold).sum() / len(true_unscaled)
    
    # --- 5. Create and Print Stats Matrix ---
    print(f"\n--- Prediction Analysis for {filepath} ---")
    
    metrics_data = {
        'run_name': filepath, 'MAE': mae, 'RMSE': rmse, 'Sign Acc': sign_accuracy,
        'Sign F1': sign_f1, f'Tol Acc (±{threshold})': tolerance_accuracy,
        f'Deadzone %': dead_zone_percent, 'T-Stat': t_stat, 'T-p_val': p_value,
        'Spearman rho': spearman_corr, 'Spearman p': spearman_p,
        'BP Stat': bp_lm_stat, 'BP p_val': bp_p_value,
    }

    metrics_df = pd.DataFrame([metrics_data])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(metrics_df.to_string(index=False, float_format='{:.4g}'.format))
    print("           Pred Neg  Pred Pos")
    print(f"True Neg {cm[0, 0]:<9} {cm[0, 1]:<9}")
    print(f"True Pos {cm[1, 0]:<9} {cm[1, 1]:<9}")

    # --- 7. Create Plots ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative (0)', 'Predicted Positive (1)'],
                yticklabels=['True Negative (0)', 'True Positive (1)'])
    plt.xlabel('Predicted Sign')
    plt.ylabel('True Sign')
    plt.title(f'Sign Prediction Confusion Matrix\n({filepath})')
    cm_plot_filename = filepath.replace('.csv', '_confusion_matrix.png')
    plt.savefig(cm_plot_filename)
    plt.clf()

    plt.figure(figsize=(10, 10))
    min_val = min(true_unscaled.min(), pred_unscaled.min())
    max_val = max(true_unscaled.max(), pred_unscaled.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label="Perfect Prediction (y=x)")
    plt.scatter(true_unscaled, pred_unscaled, alpha=0.3, label="Model Predictions")
    plt.xlabel("True Unscaled Values")
    plt.ylabel("Predicted Unscaled Values")
    plt.title(f"True vs. Predicted Regression Analysis\n({filepath})")
    plt.legend()
    plt.grid(True)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plot_filename = filepath.replace('.csv', '_analysis_plot.png')
    plt.savefig(plot_filename)
    plt.clf()
    
    return True # Return True to signal success


def main():
  
    run_name1 = sys.argv[1]
    csv_filename1 = f"out/{run_name1}.csv"

    print(f"--- Starting analysis for {run_name1} ---")

    # --- 1. Load and Analyze Source Data Stats (Only need to do this once) ---
    filename = 'qm9_filtered.npy'
    df = npy_preprocessor(filename)
    
    df['rotation'] = df['rotation'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    rotation_cols = ['rotation_633nm', 'rotation_589nm', 'rotation_335nm']
    rotation_df = pd.DataFrame(df['rotation'].to_list(), columns=rotation_cols, index=df.index)
    df = pd.concat([df, rotation_df], axis=1)

    whole_population_df = df
    chiral_mask = df['chiral_centers'].apply(len) == 1
    chiral_subpopulation_df = df[chiral_mask].copy()

    print(f"Total samples: {len(whole_population_df)}")
    print(f"Subpopulation (len==1) samples: {len(chiral_subpopulation_df)}")

    target_column = 'rotation_589nm' 
    print_stats(whole_population_df[target_column], f"Whole Population ({target_column})")
    print_stats(chiral_subpopulation_df[target_column], f"Subpopulation (len==1) ({target_column})")

    # --- 2. Generate Overlayed Histogram ---
    plt.figure(figsize=(12, 8))
    global_min = whole_population_df[target_column].min()
    global_max = whole_population_df[target_column].max()
    max_abs_val = max(abs(global_min), abs(global_max), 1)
    bins = np.geomspace(1, max_abs_val, 200)
    bins = np.concatenate([-bins[::-1], [0], bins])

    plt.hist(whole_population_df[target_column], bins=bins, alpha=0.5, label=f"Whole Population (n={len(whole_population_df)})", color='gray')
    plt.hist(chiral_subpopulation_df[target_column], bins=bins, alpha=0.8, label=f"Subpopulation (len==1) (n={len(chiral_subpopulation_df)})", color='blue')

    plt.xscale('symlog')
    plt.yscale('log')
    plt.xlabel(f"Rotation Value ({target_column})")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f"Stratified Distribution of {target_column} (Whole vs. Subpopulation)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_filename = os.path.join(OUT_DIR, f"strata_overlay_{run_name1}.png")

    plt.savefig(plot_filename)
    plt.clf()
    print(f"\nSaved stratified overlay plot to {plot_filename}")
    
    print(f" INDIVIDUAL ANALYSIS: {run_name1} ")

    if not analyze_prediction(csv_filename1, threshold=2):
        sys.exit(f"Failed to analyze {csv_filename1}. Exiting.")
    

if __name__ == '__main__':
    main()


