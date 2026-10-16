
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from core.augmentation import reflect_molecule
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import secrets
import time
import sys
import os 
import json 


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.utils import npy_preprocessor, scale_x_coordinates
from core.dataset import MoleculeSequenceDataset, QMDataModule
from core.augmentation import reflect_molecule
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_xgb")


def main():

    policy_path = sys.argv[1] 
    config = json.load(open(policy_path))
    np.random.seed(config['seed']) 
    TASK = config['TASK']
    run_name = f"{secrets.token_hex(4)}"
    print("run_name: ", run_name)
    print("config \n", config)
    start_time  = time.time()
    df_full = npy_preprocessor("qm9_filtered.npy")
    original_count = len(df_full)
    print(f"Original file has {original_count} total samples.")
    df_full = df_full.drop_duplicates(subset=['inchi'], keep='first')
    unique_count = len(df_full)
    duplicates_removed = original_count - unique_count
    print(f"Found and removed {duplicates_removed} duplicate InChI molecules.")
    print(f"There are now {unique_count} unique samples remaining.")



    if TASK == 1:
        print("TASK 1 active: Filtering population *before* splitting.")
        chiral_mask = df_full['chiral_centers'].apply(len) == 1
        
        # This is the (e.g., 20k) data we will split for train/val/test
        df_for_split = df_full[chiral_mask].copy()
        
        # This is the (e.g., 110k) data we will add to the training set
        df_scraps = df_full[~chiral_mask].copy()
        
        print(f"Separated {len(df_for_split)} chiral samples for splitting.")
        print(f"Separated {len(df_scraps)} achiral scraps for training.")
    
    else:
        # If not TASK 1, we split the whole dataset and there are no scraps
        df_for_split = df_full
        df_scraps = pd.DataFrame(columns=df_full.columns) # Empty placeholder

    
    train_val_df, test_df = train_test_split(df_for_split, test_size=0.2, random_state=43)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=43)

    # This is the (chiral-only) data we will augment
    train_chiral_df_for_aug = train_df.copy()

    if config['only_mask']: 
        real_train_df = train_df
    else:
        print(f"Adding {len(df_scraps)} recycled scraps to training set...")
        real_train_df = pd.concat([train_df, df_scraps], ignore_index=False)
        print(f"Total non-augmented training set size: {len(real_train_df)}")


    print("--- Running Data Leak Check ---")

    train_ids = set(real_train_df.index) # Contains original train + scraps
    val_ids = set(val_df.index)          # Contains original val
    test_ids = set(test_df.index)        # Contains original test

    if (leak_tv := len(train_ids & val_ids)) > 0:
        raise SystemExit(f"INDEX LEAK: {leak_tv} samples overlap between Train and Val sets. Halting.")
    if (leak_tt := len(train_ids & test_ids)) > 0:
        raise SystemExit(f"INDEX LEAK: {leak_tt} samples overlap between Train and Test sets. Halting.")
    if (leak_vt := len(val_ids & test_ids)) > 0:
        raise SystemExit(f"INDEX LEAK: {leak_vt} samples overlap between Val and Test sets. Halting.")
    print("--- LEAK CHECK (INDEX) PASSED: No index overlap found. ---")

    # --- 2. InChI Leak Check (Checks for identical molecules) ---
    print("--- Running InChI Leak Check ---")
    
    # Get the InChI strings from the 'inchi' column of each dataframe
    train_inchis = set(real_train_df['inchi'])
    val_inchis = set(val_df['inchi'])
    test_inchis = set(test_df['inchi'])

    if (leak_tv_inchi := len(train_inchis & val_inchis)) > 0:
        raise SystemExit(f"INCHI LEAK: {leak_tv_inchi} molecules overlap between Train and Val sets. Halting.")
    if (leak_tt_inchi := len(train_inchis & test_inchis)) > 0:
        raise SystemExit(f"INCHI LEAK: {leak_tt_inchi} molecules overlap between Train and Test sets. Halting.")
    if (leak_vt_inchi := len(val_inchis & test_inchis)) > 0:
        raise SystemExit(f"INCHI LEAK: {leak_vt_inchi} molecules overlap between Val and Test sets. Halting.")
    
    print("--- LEAK CHECK (INCHI) PASSED: No molecular identity overlap found. ---")

    y_scale_df = ((np.stack(real_train_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    y_scaler = RobustScaler()
    y_scaler.fit(y_scale_df) 
    # --- X Scaling ---
    # Fit scaler ONLY on the original, non-aug train data
    X_train_coords_flat = np.concatenate(list(train_df['xyz'].values))[:, :3]
    x_coord_scaler = StandardScaler()
    x_coord_scaler.fit(X_train_coords_flat) 

    # -------------------------------------------------------------------------
    # 1. Chiral Reflection Augmentation
    # -------------------------------------------------------------------------
    if config.get('use_reflection', False):
        print("Applying chiral reflection augmentation...")
        X_chiral = np.stack(train_chiral_df_for_aug['xyz'].values)
        y_chiral = ((np.stack(train_chiral_df_for_aug['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
        
        X_reflected = [reflect_molecule(m) for m in tqdm(X_chiral, desc="Applying Chiral Reflection")]
        train_aug1 = pd.DataFrame({
            'xyz': X_reflected,
            'rotation': [np.array([0, val[0] * -1, 0]) for val in y_chiral]
        })
    else:
        train_aug1 = pd.DataFrame(columns=['xyz', 'rotation'])

    # -------------------------------------------------------------------------
    # 2. Rotation & Translation Augmentation via MoleculeSequenceDataset
    # -------------------------------------------------------------------------
    if config.get('augment', False):
        augment_passes = config.get('augment_size', 1)
        print(f"Applying rotate/translate augmentation {augment_passes} times via MoleculeSequenceDataset...")

        # Instantiate the untouched dataset
        aug_dataset = MoleculeSequenceDataset(
            X=train_df['xyz'].values,
            y=train_df['rotation'].values,
            augment=True,
            reflection_only=False,
            shuffle_atoms=config.get('shuffle_atoms', False)
        )

        X_aug_list = []
        y_aug_list = []

        # Iterate through dataset; each call runs dynamic __getitem__ transformations
        for aug_pass in range(augment_passes):
            desc = f"Augment Pass {aug_pass + 1}/{augment_passes}"
            for i in tqdm(range(len(aug_dataset)), desc=desc):
                mol_tensor, rot_tensor = aug_dataset[i]
                # Convert PyTorch tensors back to NumPy arrays for pandas & XGBoost
                X_aug_list.append(mol_tensor.numpy())
                y_aug_list.append(rot_tensor.numpy())

        train_aug2 = pd.DataFrame({
            'xyz': X_aug_list,
            'rotation': y_aug_list
        })
        print(f"Created {len(train_aug2)} new rotate/translate augmentations.")
    else:
        train_aug2 = pd.DataFrame(columns=['xyz', 'rotation'])

    # Combine base train set + reflection augmentations + dataset augmentations
    raw_combined_train_df = pd.concat([real_train_df, train_aug1, train_aug2], ignore_index=True)

    y_train_raw = np.stack(raw_combined_train_df['rotation'].values)[:, 1].astype(float)
    
    mean_orig = np.mean(y_train_raw)
    std_orig = np.std(y_train_raw)
    z_scores_orig = (y_train_raw - mean_orig) / std_orig

    # ----------------------------------------------------
    # 3. Create Inlier Mask & Filter DataFrame
    # ----------------------------------------------------
    z_thresh = config.get('train_omit_z_score')
    
    # Keep only samples within the Z-score boundary (|Z| <= threshold)
    inlier_mask = np.abs(z_scores_orig) <= z_thresh
    outlier_count = int((~inlier_mask).sum())
    print("outlier_count", outlier_count)
    final_train_df = raw_combined_train_df[inlier_mask].reset_index(drop=True)

    # Our val and test sets remain the "clean" originals
    final_val_df = val_df
    final_test_df = test_df

    test_ids_to_pass = final_test_df.index.values
    print(f"Final Train set size (with aug): {len(final_train_df)}")
    print(f"Final Val set size: {len(final_val_df)}")
    print(f"Final Test set size: {len(final_test_df)}")

    # --- Extract X, y arrays from the FINAL dataframes ---
    X_train = list(final_train_df['xyz'].values)
    y_train = ((np.stack(final_train_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    
    X_val = list(final_val_df['xyz'].values)
    y_val = ((np.stack(final_val_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    
    X_test = list(final_test_df['xyz'].values)
    y_test = ((np.stack(final_test_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))


    X_train_scaled = scale_x_coordinates(X_train, x_coord_scaler)
    X_val_scaled = scale_x_coordinates(X_val, x_coord_scaler)
    X_test_scaled = scale_x_coordinates(X_test, x_coord_scaler)

  
    y_train_scaled = y_scaler.transform(y_train).flatten()
    y_val_scaled = y_scaler.transform(y_val).flatten()
    y_test_scaled = y_scaler.transform(y_test).flatten()

    X_train_3d = np.stack(X_train_scaled)
    X_val_3d = np.stack(X_val_scaled)
    X_test_3d = np.stack(X_test_scaled)
    
    S, F = config['patch_size'], config['in_channels']

    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], S * F)
    X_val_flat = X_val_3d.reshape(X_val_3d.shape[0], S * F)
    X_test_flat = X_test_3d.reshape(X_test_3d.shape[0], S * F)
    
    y_train_flat = y_train_scaled
    y_val_flat = y_val_scaled

    print("--- Training XGBoost Regressor ---")
    if config['criterion'] == 'huber':
        criterion = 'reg:pseudohubererror'  
    else:
        criterion = 'reg:squarederror'
    xgb_params = {
        'objective': criterion,
        'eval_metric': config['eval_metric'],
        'eta': config['learning_rate'],  
        'n_estimators': config['epochs'], 
        'max_depth': config['max_depth'],
        'n_jobs': config['n_jobs'],
        'seed': config['seed'], 
        'reg_lambda': config['reg_lambda'],
        'early_stopping_rounds': config['patience'],
        'tree_method':"hist",  
        'device':"cuda",       
    }
    model = xgb.XGBRegressor(**xgb_params)
    
    model.fit(
        X_train_flat, 
        y_train_flat, 
        eval_set=[(X_val_flat, y_val_flat)],
        verbose=config['epochs'] - 1  # True#
    )
    # Final prediction and metric calculation
    scaled_preds = model.predict(X_test_flat)
    y_test_scaled_2d = y_test_scaled.reshape(-1, 1) # Reshape 1D scaled array to 2D

    # FIX LINE 1: Use the scaled true labels array for inverse_transform
    unscaled_labels = y_scaler.inverse_transform(y_test_scaled_2d).flatten() 
    
    unscaled_preds = y_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()

    results_df = pd.DataFrame({
        'item_id': test_ids_to_pass,
        'true_value_unscaled': unscaled_labels.flatten(),
        'prediction_unscaled': unscaled_preds.flatten(),
        'true_scaled': y_test_scaled.flatten(),
        'pred_scaled': scaled_preds.flatten(),
    })
    
    # 2. Add the Unscaled Error
    results_df['unscaled_error'] = np.abs(results_df['true_value_unscaled'] - results_df['prediction_unscaled'])

    os.makedirs(OUT_DIR, exist_ok=True)
            # Use the wandb run name to make the file unique
    csv_filename = f"{run_name}.csv"
    output_file = os.path.join(OUT_DIR, csv_filename)


    
    results_df.to_csv(output_file, index=False)
    print(f"\nFull prediction results saved to {output_file}")


    # Reset display options
    pd.reset_option('display.float_format')
    final_mae = mean_absolute_error(unscaled_labels, unscaled_preds)
    
    print(f"\nFinal MAE: {final_mae:.4f}")

    end_time = time.time()
    print(end_time - start_time)
if __name__ == "__main__":
    main()
