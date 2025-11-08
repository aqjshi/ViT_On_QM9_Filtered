import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
from scipy import stats
import statsmodels.api as sm
import seaborn as sns

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df

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

def compare_models(filepath1, filepath2):
    """
    Compares two model prediction CSVs on their intersecting item_ids.
    """
    print("\n\n--- Intersection Performance Comparison ---")
    
    try:
        df1 = pd.read_csv(filepath1)
        df2 = pd.read_csv(filepath2)
    except FileNotFoundError as e:
        print(f"Error loading file for comparison: {e}")
        return

    # Extract run names for cleaner labels
    model1_name = filepath1.replace('.csv', '')
    model2_name = filepath2.replace('.csv', '')

    # Find intersection using a merge
    merged_df = pd.merge(df1, df2, on='item_id', suffixes=('_m1', '_m2'))
    
    num_intersection = len(merged_df)
    if num_intersection == 0:
        print("Error: No intersecting item_ids found between the two files. Cannot compare.")
        return
        
    print(f"Found {num_intersection} intersecting samples for comparison.")

    # Sanity Check: Verify true values are identical
    is_equal = (merged_df['true_value_unscaled_m1'] == merged_df['true_value_unscaled_m2']).all()
    if not is_equal:
        print("WARNING: 'true_value_unscaled' for intersecting items are NOT identical. Comparison may be invalid.")
    else:
        print("Sanity Check: 'true_value_unscaled' for intersecting items are identical. Proceeding.")

    # --- Calculate Metrics on Intersecting Set ---
    y_true = merged_df['true_value_unscaled_m1'] # Use m1 as the "true" source
    y_true_class = (y_true > 0).astype(int)
    
    # Model 1 Metrics
    y_pred_m1 = merged_df['prediction_unscaled_m1']
    mae_m1 = mean_absolute_error(y_true, y_pred_m1)
    rmse_m1 = np.sqrt(mean_squared_error(y_true, y_pred_m1))
    y_pred_class_m1 = (y_pred_m1 > 0).astype(int)
    sign_acc_m1 = accuracy_score(y_true_class, y_pred_class_m1)
    sign_f1_m1 = f1_score(y_true_class, y_pred_class_m1)

    # Model 2 Metrics
    y_pred_m2 = merged_df['prediction_unscaled_m2']
    mae_m2 = mean_absolute_error(y_true, y_pred_m2)
    rmse_m2 = np.sqrt(mean_squared_error(y_true, y_pred_m2))
    y_pred_class_m2 = (y_pred_m2 > 0).astype(int)
    sign_acc_m2 = accuracy_score(y_true_class, y_pred_class_m2)
    sign_f1_m2 = f1_score(y_true_class, y_pred_class_m2)

    # --- Create Comparison Matrix ---
    data = {
        'Metric': ['MAE', 'RMSE', 'Sign Acc', 'Sign F1'],
        model1_name: [mae_m1, rmse_m1, sign_acc_m1, sign_f1_m1],
        model2_name: [mae_m2, rmse_m2, sign_acc_m2, sign_f1_m2],
    }
    
    comp_df = pd.DataFrame(data)
    
    # Determine winner for each metric
    winners = []
    for i, row in comp_df.iterrows():
        metric_name = row['Metric']
        val1 = row[model1_name]
        val2 = row[model2_name]
        
        # Lower is better for MAE/RMSE
        if metric_name in ['MAE', 'RMSE']:
            if val1 < val2:
                winners.append(f"{model1_name} wins")
            elif val2 < val1:
                winners.append(f"{model2_name} wins")
            else:
                winners.append('Tie')
        # Higher is better for Sign Acc/F1
        else:
            if val1 > val2:
                winners.append(f"{model1_name} wins")
            elif val2 > val1:
                winners.append(f"{model2_name} wins")
            else:
                winners.append('Tie')
                
    comp_df['Winner'] = winners
    
    print("\n--- Comparison on Intersecting Samples ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comp_df.to_string(index=False, float_format='{:.4g}'.format))

def main():
    if len(sys.argv) < 3: # Check for script name + 2 run names
        print("ERROR: Not enough run names provided.")
        print("Usage: python analysis.py <RUN_NAME_1> <RUN_NAME_2>")
        sys.exit(1) # Exit with an error code
    
    run_name1 = sys.argv[1]
    csv_filename1 = f"{run_name1}.csv"
    run_name2 = sys.argv[2]
    csv_filename2 = f"{run_name2}.csv" # <-- Fixed bug

    print(f"--- Starting analysis for {run_name1} and {run_name2} ---")

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

    plot_filename = f"strata_overlay_{run_name1}_vs_{run_name2}.png" # Generic name
    plt.savefig(plot_filename)
    plt.clf()
    print(f"\nSaved stratified overlay plot to {plot_filename}")
    
    # --- 3. Run Prediction Analysis for each file ---
    print("\n" + "="*50)
    print(f" INDIVIDUAL ANALYSIS: {run_name1} ".center(50, "="))
    print("="*50)
    if not analyze_prediction(csv_filename1, threshold=2):
        sys.exit(f"Failed to analyze {csv_filename1}. Exiting.")
    
    print("\n" + "="*50)
    print(f" INDIVIDUAL ANALYSIS: {run_name2} ".center(50, "="))
    print("="*50)
    if not analyze_prediction(csv_filename2, threshold=2):
        sys.exit(f"Failed to analyze {csv_filename2}. Exiting.")

    # --- 4. Run Head-to-Head Comparison ---
    print("\n" + "="*50)
    print(f" H2H COMPARISON: {run_name1} vs {run_name2} ".center(50, "="))
    print("="*50)
    compare_models(csv_filename1, csv_filename2)

if __name__ == '__main__':
    main()


# 
