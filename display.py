import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
import sys  # Make sure sys is imported
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import statsmodels.api as sm  # <-- NEW IMPORT
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix # <-- MODIFIED
import seaborn as sns                                                  # <-- NEW
def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df

def print_stats(series, name):
    """Helper function to print descriptive statistics."""
    print(f"\n--- Statistics for {name} ---")
    
    # Check if series is empty to avoid errors
    if series.empty:
        print("  No data in this stratum.")
        return

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

    # Calculate and print outlier counts based on 1.5*IQR rule
    clip_min = series.quantile(0.25) - (1.5 * iqr)
    clip_max = series.quantile(0.75) + (1.5 * iqr)
    outliers = (series < clip_min) | (series > clip_max)
    print(f"  Outliers (1.5*IQR): {outliers.sum()} ({outliers.sum() / series.count() * 100:.2f}%)")



def analyze_prediction(filepath, threshold=0.5):
    """
    Loads a prediction CSV and performs a full regression analysis.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"\n--- Analysis Failed: Could not find file '{filepath}' ---")
        print("Did the trainer.test() step run successfully?")
        # Return a tuple of Nones matching the expected output
        return (None,) * 8 # Updated to 8 return values

    true_unscaled = df['true_value_unscaled']
    pred_unscaled = df['prediction_unscaled']
    
    # --- 1. Calculate Standard REGRESSION Metrics ---
    mae = mean_absolute_error(true_unscaled, pred_unscaled)
    mse = mean_squared_error(true_unscaled, pred_unscaled)
    rmse = np.sqrt(mse)
    
    # --- 2. Perform Statistical Test (Paired T-test) ---
    t_stat, p_value = stats.ttest_rel(true_unscaled, pred_unscaled)
    
    print(f"\n--- Prediction Analysis for {filepath} ---")
    print(f"--- REGRESSION METRICS ---")
    print(f"  MAE (Unscaled):   {mae:.4f}")
    print(f"  RMSE (Unscaled):  {rmse:.4f}")
    print(f"  Paired T-test (True vs. Pred):")
    print(f"    T-statistic:  {t_stat:.4f}")
    print(f"    P-value:      {p_value:.4g} (Small p-value suggests systematic bias)")

    # --- 3. HETEROSCEDASTICITY ANALYSIS ---
    errors = true_unscaled - pred_unscaled
    abs_errors = np.abs(errors)
    squared_errors = errors**2
    
    spearman_corr, spearman_p = stats.spearmanr(pred_unscaled, abs_errors)
    print(f"  Spearman's Rank (Pred vs. Abs(Error)):")
    print(f"    Correlation:  {spearman_corr:.4f} (Positive value = 'cone of shame')")
    print(f"    P-value:      {spearman_p:.4g}")

    exog = sm.add_constant(pred_unscaled) # Add an intercept
    bp_test = sm.stats.het_breuschpagan(squared_errors, exog)
    print(f"  Breusch-Pagan Test (Pred vs. Squared(Error)):")
    print(f"    LM Statistic: {bp_test[0]:.4f}")
    print(f"    P-value:      {bp_test[1]:.4g} (Small p-value suggests heteroscedasticity)")

    # --- 4. CLASSIFICATION & TOLERANCE METRICS ---
    print(f"\n--- CLASSIFICATION METRICS (as Regression) ---")

    # Metric 1: Sign Prediction (Positive vs. Negative)
    y_true_class = (true_unscaled > 0).astype(int) # 1=Pos, 0=Neg
    y_pred_class = (pred_unscaled > 0).astype(int) # 1=Pos, 0=Neg
    
    sign_accuracy = accuracy_score(y_true_class, y_pred_class)
    sign_f1 = f1_score(y_true_class, y_pred_class, average='binary') # Use 'binary' for 2 classes
    
    print(f"  Sign Prediction (True/Pred > 0):")
    print(f"    Sign Accuracy: {sign_accuracy:.4f}")
    print(f"    Sign F1-Score: {sign_f1:.4f}")

    # --- START OF NEW CODE BLOCK ---
    
    # Calculate and print the confusion matrix
    cm = confusion_matrix(y_true_class, y_pred_class)
    print("    Confusion Matrix (Sign Prediction):")
    print("       Pred Neg  Pred Pos")
    print(f"True Neg {cm[0, 0]:<9} {cm[0, 1]:<9}")
    print(f"True Pos {cm[1, 0]:<9} {cm[1, 1]:<9}")

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative (0)', 'Predicted Positive (1)'],
                yticklabels=['True Negative (0)', 'True Positive (1)'])
    plt.xlabel('Predicted Sign')
    plt.ylabel('True Sign')
    plt.title('Sign Prediction Confusion Matrix')
    
    # Save the plot
    cm_plot_filename = filepath.replace('.csv', '_confusion_matrix.png')
    plt.savefig(cm_plot_filename)
    plt.clf() # Clear the plot
    print(f"    Saved confusion matrix plot to {cm_plot_filename}")
    
    # --- END OF NEW CODE BLOCK ---

    # Metric 2: "Dead Zone" / Tolerance Metrics
    dead_zone_count = (np.abs(true_unscaled) <= threshold).sum()
    dead_zone_percent = dead_zone_count / len(true_unscaled)
    
    tolerance_accuracy = (abs_errors <= threshold).sum() / len(true_unscaled)
    
    print(f"\n  Tolerance Metrics (Threshold = +/- {threshold}):")
    print(f"    Data in 'Dead Zone': {dead_zone_count} ({dead_zone_percent * 100:.2f}%)")
    print(f"    Tolerance Accuracy:  {tolerance_accuracy:.4f} (% of preds within +/- {threshold} of true)")

    # --- 5. Create Comparison Graph (True vs. Predicted Plot) ---
    plt.figure(figsize=(10, 10))
    
    min_val = min(true_unscaled.min(), pred_unscaled.min())
    max_val = max(true_unscaled.max(), pred_unscaled.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', linestyle='--', 
             linewidth=2, label="Perfect Prediction (y=x)")
             
    plt.scatter(true_unscaled, pred_unscaled, alpha=0.3, label="Model Predictions")
    
    plt.xlabel("True Unscaled Values")
    plt.ylabel("Predicted Unscaled Values")
    plt.title("True vs. Predicted Regression Analysis")
    plt.legend()
    plt.grid(True)
    
    plt.xscale('symlog')
    plt.yscale('symlog')
    
    plot_filename = filepath.replace('.csv', '_analysis_plot.png')
    plt.savefig(plot_filename)
    plt.clf()
    print(f"  Saved analysis plot to {plot_filename}")
    
    # Return all the new metrics
    return mae, rmse, sign_accuracy, sign_f1, tolerance_accuracy, cm, spearman_corr, bp_test[1]



def main():

    # strata 1 (whole population, subpopulation where chiral_centers ==1) * (rotation), overlay histograms on jyust rotation
 
    # end result: 
    # 1 ( mean median iqr min max ) *(whole population, subpopulation where chiral_centers ==1)
    # 2 histogram showing rotation values for whole, sub populations overlay

    filename = 'qm9_filtered.npy'
    df = npy_preprocessor(filename)
    
    # --- 1. Data Prep ---
    df['rotation'] = df['rotation'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    rotation_cols = ['rotation_633nm', 'rotation_589nm', 'rotation_335nm']
    rotation_df = pd.DataFrame(df['rotation'].to_list(), columns=rotation_cols, index=df.index)
    df = pd.concat([df, rotation_df], axis=1)

    # --- 2. Create Strata ---
    # Stratum 1: Whole population
    whole_population_df = df
    
    # Stratum 2: Subpopulation (chiral_centers == 1)
    chiral_mask = df['chiral_centers'].apply(len) == 1
    chiral_subpopulation_df = df[chiral_mask].copy() # Use .copy() to avoid warnings

    print(f"Total samples: {len(whole_population_df)}")
    print(f"Subpopulation (len==1) samples: {len(chiral_subpopulation_df)}")

    # --- 3. Print Statistics for both strata ---
    # We use rotation_589nm as it's the main target
    target_column = 'rotation_589nm' 
    
    print_stats(whole_population_df[target_column], f"Whole Population ({target_column})")
    print_stats(chiral_subpopulation_df[target_column], f"Subpopulation (len==1) ({target_column})")

    # --- 4. Generate Overlayed Histogram ---
    plt.figure(figsize=(12, 8))

    # Use the global min/max for a consistent bin range
    global_min = whole_population_df[target_column].min()
    global_max = whole_population_df[target_column].max()
    
    # Create symmetrical log bins to handle 0 and outliers
    max_abs_val = max(abs(global_min), abs(global_max), 1) # Ensure max_abs_val is at least 1
    bins = np.geomspace(1, max_abs_val, 200)
    bins = np.concatenate([-bins[::-1], [0], bins])

    # Plot the Whole Population
    plt.hist(whole_population_df[target_column], 
             bins=bins, 
             alpha=0.5, 
             label=f"Whole Population (n={len(whole_population_df)})", 
             color='gray')
             
    # Plot the Subpopulation (len==1) on top
    plt.hist(chiral_subpopulation_df[target_column], 
             bins=bins, 
             alpha=0.8, 
             label=f"Subpopulation (len==1) (n={len(chiral_subpopulation_df)})", 
             color='blue')

    plt.xscale('symlog') # Symmetrical Log Scale is a must
    plt.yscale('log')    # Log scale for frequency
    
    plt.xlabel(f"Rotation Value ({target_column})")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f"Stratified Distribution of {target_column} (Whole vs. Subpopulation)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plot_filename = "rotation_strata_1_overlay.png"
    plt.savefig(plot_filename)
    plt.clf()
    print(f"\nSaved stratified overlay plot to {plot_filename}")
    threshold =67.94521
    mae, rmse, sign_accuracy, sign_f1, tolerance_accuracy, cm, spearman_corr, bp_test = analyze_prediction("REGRESSION_TASK0_test_predictions.csv", threshold)
if __name__ == '__main__':
    main()
