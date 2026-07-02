
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from augmentation import translate_molecule, rotate_molecule, reflect_molecule
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
import secrets
import time

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df

def main():
    RANDOM_STATE = 43
    np.random.seed(RANDOM_STATE) # Set NumPy seed for repeatability
    optimal_config_values = {
        'TASK': 1,
        'augment': False,
        'only_mask': False,
        'use_reflection': True, 
        'augment_size': 2,   
        'seq_len': 27,          
       'input_feature_len': 8,   
        'eval_metric': 'mae',
        'eta':.02,
        'n_estimators': 6000,
        'max_depth':16, # Assuming a default sensible depth
        'n_jobs': 6, 
        'seed': RANDOM_STATE,
        'reg_lambda': 4

    } 
    print('eta', optimal_config_values)
    TASK = optimal_config_values['TASK']

    run_name = f"{secrets.token_hex(4)}"
    start_time  = time.time()
    df_full = npy_preprocessor("qm9_filtered.npy")
    original_count = len(df_full)
    print(f"Original file has {original_count} total samples.")
    df_full = df_full.drop_duplicates(subset=['inchi'], keep='first')
    unique_count = len(df_full)
    duplicates_removed = original_count - unique_count
    print(f"Found and removed {duplicates_removed} duplicate InChI molecules.")
    print(f"There are now {unique_count} unique samples remaining.")



    chiral_mask = df_full['chiral_centers'].apply(len) == 1
    if TASK == 1:
        print("TASK 1 active: Filtering population *before* splitting.")

        
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

    
    train_val_df, test_df = train_test_split(df_for_split, test_size=0.2, random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=RANDOM_STATE)

    if optimal_config_values['only_mask']: 
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

    if optimal_config_values['use_reflection']:
        
        # This is the (chiral-only) data we will augment
        train_chiral_df_for_aug = train_df[chiral_mask].copy()

        print("Applying chiral reflection augmentation...")
        # We use the *chiral-only* `train_chiral_df_for_aug` we saved earlier
        X_chiral = np.stack(train_chiral_df_for_aug['xyz'].values)
        y_chiral = ((np.stack(train_chiral_df_for_aug['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
        
        X_reflected = []
        for i in tqdm(range(len(X_chiral)), desc="Applying Chiral Reflection"):
            X_reflected.append(reflect_molecule(X_chiral[i]))
        
        train_aug1 = pd.DataFrame({
            'xyz': X_reflected,
            'rotation': [np.array([0, val[0] * -1, 0]) for val in y_chiral]
        })
    else:
        train_aug1 = pd.DataFrame(columns=['xyz', 'rotation'])

    if optimal_config_values['augment']:
        print(f"Applying rotate/translate augmentation {optimal_config_values['augment_size']} times...")
        
        # Get the base data *once*
        X_to_aug = np.stack(train_df['xyz'].values)
        y_to_aug_base = train_df['rotation'].values # Get the original 'rotation' column
        
        # Master lists to hold all augmented samples
        X_augmented_master = []
        y_augmented_master = []

        # Your new loop
        for aug_pass in range(optimal_config_values['augment_size']):
            
            desc = f"Augment Pass {aug_pass + 1}/{optimal_config_values['augment_size']}"
            for i in tqdm(range(len(X_to_aug)), desc=desc):
                molecule = X_to_aug[i]
                aug_choice = np.random.choice(['rotate', 'translate'])
                
                if aug_choice == 'rotate':
                    angle = np.random.uniform(0, 2 * np.pi)
                    axis = np.random.choice(['x', 'y', 'z'])
                    molecule = rotate_molecule(molecule, angle, axis=axis)
                elif aug_choice == 'translate':
                    molecule = translate_molecule(molecule, magnitude=0.02)
                
                # Append the single augmented molecule to the master list
                X_augmented_master.append(molecule)
                # Append its corresponding (unchanged) label
                y_augmented_master.append(y_to_aug_base[i])

        # Create the DataFrame *once* from the master lists
        train_aug2 = pd.DataFrame({
            'xyz': X_augmented_master,
            'rotation': y_augmented_master  # The target does not change
        })
        print(f"Created {len(train_aug2)} new rotate/translate augmentations.")

    else:
        train_aug2 = pd.DataFrame(columns=['xyz', 'rotation'])

    print(f"Created {len(train_aug2)} new rotate/translate augmentations.")
    final_train_df = pd.concat([real_train_df, train_aug2], ignore_index=True)

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


    
    def scale_x_coordinates(X_split, scaler):
        if len(X_split) == 0:
            return []
        X_stacked = np.stack(X_split) 
        coords = X_stacked[:, :, :3]
        features = X_stacked[:, :, 3:]
        coords_scaled_flat = scaler.transform(coords.reshape(-1, 3))
        coords_scaled = coords_scaled_flat.reshape(X_stacked.shape[0], X_stacked.shape[1], 3)
        X_scaled_stacked = np.concatenate((coords_scaled, features), axis=2)
        return [X_scaled_stacked[i, ...] for i in range(X_scaled_stacked.shape[0])]
    
    X_train_scaled = scale_x_coordinates(X_train, x_coord_scaler)
    X_val_scaled = scale_x_coordinates(X_val, x_coord_scaler)
    X_test_scaled = scale_x_coordinates(X_test, x_coord_scaler)

  
    y_train_scaled = y_scaler.transform(y_train).flatten()
    y_val_scaled = y_scaler.transform(y_val).flatten()
    y_test_scaled = y_scaler.transform(y_test).flatten()


    X_train_3d = np.stack(X_train_scaled)
    X_val_3d = np.stack(X_val_scaled)
    X_test_3d = np.stack(X_test_scaled)
    
    S, F = optimal_config_values['seq_len'], optimal_config_values['input_feature_len']

    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0], S * F)
    X_val_flat = X_val_3d.reshape(X_val_3d.shape[0], S * F)
    X_test_flat = X_test_3d.reshape(X_test_3d.shape[0], S * F)
    
    y_train_flat = y_train_scaled
    y_val_flat = y_val_scaled

    print("--- Training XGBoost Regressor ---")

    xgb_params = {
        'objective': 'reg:pseudohubererror',
        'eval_metric': optimal_config_values['eval_metric'],
        'eta': optimal_config_values['eta'],  # Maps to learning_rate
        'n_estimators': optimal_config_values['n_estimators'], # Maps to epochs
        'max_depth': optimal_config_values['max_depth'],
        'n_jobs': optimal_config_values['n_jobs'],
        'seed': optimal_config_values['seed'], # Use the seed from the config
        'reg_lambda': optimal_config_values['reg_lambda'] # Maps to weight_decay
        ,'early_stopping_rounds': 500,
        'tree_method':"hist",  # Use the 'hist' method
        'device':"cuda",       # Tell it to use the GPU
    }
    model = xgb.XGBRegressor(**xgb_params)
    
    model.fit(
        X_train_flat, 
        y_train_flat, 
        eval_set=[(X_val_flat, y_val_flat)],
        verbose=optimal_config_values['n_estimators'] - 1  # True#
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
    
    # 3. Save the full results DataFrame to a CSV file
    results_filename = f"{run_name}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"\nFull prediction results saved to {results_filename}")


    # Reset display options
    pd.reset_option('display.float_format')
    final_mae = mean_absolute_error(unscaled_labels, unscaled_preds)
    
    print(f"\nFinal MAE: {final_mae:.4f}")

    end_time = time.time()
    print(end_time - start_time)
if __name__ == "__main__":
    main()
