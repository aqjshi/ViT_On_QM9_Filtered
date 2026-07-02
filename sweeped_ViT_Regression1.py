
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics import Accuracy
from tqdm import tqdm
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from pytorch_lightning.tuner import Tuner
import argparse 
from torchmetrics import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler, RobustScaler
from ViT import ViT
def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df





class MoleculeSequenceDataset(Dataset):
    def __init__(self, X, y, augment=False, reflection_only=False):
        self.X = X
        self.y = y
        self.augment = augment
        self.reflection_only = reflection_only 
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        molecule = self.X[idx]
        target = self.y[idx]
 
        if self.augment and not self.reflection_only:
            aug_choice = np.random.choice(['rotate', 'translate'])
            if aug_choice == 'rotate':
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.choice(['x', 'y', 'z'])
                molecule = rotate_molecule(molecule, angle, axis=axis)
            elif aug_choice == 'translate':
                molecule = translate_molecule(molecule, magnitude=0.02)

        return torch.tensor(molecule, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

def rotate_molecule(xyz_data, angle, axis='z'):
    """Rotate the first 3 columns (x, y, z) around the chosen axis by 'angle'."""
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:  # 'z'
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    rotated_coords = np.dot(coords_3d, rotation_matrix.T)  # (27,3)

    rotated_xyz_data = np.hstack((rotated_coords, other_feats))  # Still (27,8)
    return rotated_xyz_data

def translate_molecule(xyz_data, magnitude=0.001):
    """Translate the first 3 columns (x, y, z) by a random magnitude along a random axis."""
    translation_vector = np.zeros(3)
    random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
    translation_vector[random_axis] = magnitude

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    translated_coords = coords_3d + translation_vector  # Apply translation

    translated_xyz_data = np.hstack((translated_coords, other_feats))  # Still (27,8)
    return translated_xyz_data

def reflect_molecule(xyz_data):
    """Reflect the first 3 columns (x, y, z) across a randomly chosen axis."""
    reflection_matrix = np.eye(3)
    random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
    reflection_matrix[random_axis, random_axis] = -1  # Reflect across the chosen axis

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    reflected_coords = np.dot(coords_3d, reflection_matrix.T)  # Apply reflection

    reflected_xyz_data = np.hstack((reflected_coords, other_feats))  # Still (27,8)
    return reflected_xyz_data


class QMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64): 
        super().__init__()
        self.batch_size = batch_size

    def set_datasets(self, train_dataset, val_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    # dont change this, optimal performance for windows, any change in configuration will increase runtime by 15%
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=1, persistent_workers=True, pin_memory=False) #,  pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2, num_workers=0) 
    
    



class ViTModule(pl.LightningModule):
    def __init__(self, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0, scaler=None, 
                 weight_decay=0.0, test_ids=None, output_file_name= None): 
        super().__init__()
        self.save_hyperparameters()
        self.test_ids =test_ids
        self.output_file_name =output_file_name
        self.model = ViT(embedding_dim=embedding_dim, 
                         num_classes=1, 
                         embedding_dropout=embedding_dropout_rate, 
                         mlp_dropout=mlp_dropout_rate,
                         num_transformer_layers = num_transformer_layers, 
                         num_heads = num_heads,
                         mlp_size = mlp_size
                         )
        self.scaler = scaler
        self.criterion =  nn.HuberLoss(delta=1.0) 

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.OVERFLOW_CLIP_VAL = 5
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y) 
        
        self.train_mae(logits.squeeze(1), y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/scaled_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True) 
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y.float()) 

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True) 
        self.validation_step_outputs.append({'preds': logits.squeeze(1), 'labels': y}) 
        return loss
    
    def on_validation_epoch_end(self):
        # Prevent errors on sanity check
        if not self.validation_step_outputs:
            return

        all_scaled_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_scaled_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
        self.validation_step_outputs.clear() 


        scaled_mae = np.abs(all_scaled_preds - all_scaled_labels).mean()
        self.log('val/scaled_mae', scaled_mae, on_epoch=True, prog_bar=True)
        unscaled_preds = self.scaler.inverse_transform(all_scaled_preds.reshape(-1, 1)).flatten()
        unscaled_labels = self.scaler.inverse_transform(all_scaled_labels.reshape(-1, 1)).flatten()
        
        # This is now the correct unscaled MAE
        unscaled_mae = np.abs(unscaled_preds - unscaled_labels).mean()
        self.log('val/mae', unscaled_mae, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        # 1. Gather 1D scaled arrays
        all_scaled_preds_flat = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_scaled_labels_flat = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        self.test_step_outputs.clear() 

        # 2. CALCULATE SCALED MAE (Correct)
        scaled_mae = np.abs(all_scaled_preds_flat - all_scaled_labels_flat).mean()
        
        self.log('test/scaled_mae', scaled_mae, on_epoch=True, prog_bar=True)
        
        # 3. FIX: Reshape the 1D arrays to 2D (n_samples, 1) for inverse_transform
        scaled_preds_2d = all_scaled_preds_flat.reshape(-1, 1)
        scaled_labels_2d = all_scaled_labels_flat.reshape(-1, 1)

        # 4. CALCULATE UNSCALED MAE
        unscaled_preds = self.scaler.inverse_transform(scaled_preds_2d).flatten() # Output is (N, 1), flatten back to 1D
        unscaled_labels = self.scaler.inverse_transform(scaled_labels_2d).flatten() # Output is (N, 1), flatten back to 1D
        
        unscaled_mae = np.abs(unscaled_preds - unscaled_labels).mean()
        
        self.log('test/mae', unscaled_mae, on_epoch=True, prog_bar=True)
        results_df = pd.DataFrame({
            'item_id': self.test_ids,
            'true_value_unscaled': unscaled_labels,
            'prediction_unscaled': unscaled_preds, 
            'Unscaled Error': np.abs(unscaled_preds - unscaled_labels),
            'true_value_scaled': all_scaled_labels_flat, 
            'prediction_scaled': all_scaled_preds_flat, 
        })
        # Use the wandb run name to make the file unique
        csv_filename = f"{self.output_file_name}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"Saved predictions to {csv_filename}")

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
    
        loss = self.criterion(logits.squeeze(1), y) 
        
        self.test_mae(logits.squeeze(1), y) 
        
        self.log('test/loss', loss, on_epoch=True)
        
        self.log('test/scaled_mae', self.test_mae, on_epoch=True, prog_bar=True) 
        
        self.test_step_outputs.append({'preds': logits.squeeze(1), 'labels': y}) # Also remove squeeze(1) here
        return loss
            

    def configure_optimizers(self):
        LR_START = 0.00015       # The initial learning rate (self.hparams.learning_rate)
        E_P1 = 70                # End of Phase 1 (Start of Aggressive Decay)
        E_P2 = 200                # End of Phase 2 (Start of Fine-Tuning Decay)
        LR_P2_END = 0.00006      # Target LR at Epoch 40
        LR_END = 0.0000001        # Final target LR at the end of total epochs
        
        # --- CALCULATIONS ---
        EPOCHS_TOTAL = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 150

        # The end factor for the final LR, relative to LR_START
        FINAL_END_FACTOR = LR_END / LR_START 
        
        # The end factor for the Phase 2 LR, relative to LR_START
        FACTOR_P2_END = LR_P2_END / LR_START
        
        # --- OPTIMIZER ---
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                    lr=LR_START, 
                                    weight_decay=self.hparams.weight_decay, 
                                    eps=1e-7, 
                                    betas=(0.8, 0.99))

        # ----------------------------------------------------
        # Phase 1: Constant LR (Epoch 0 to E_P1=10)
        # ----------------------------------------------------
        # total_iters = 10
        scheduler1 = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=1.0, 
            total_iters=E_P1 
        )
        
        # ----------------------------------------------------
        # Phase 2: Aggressive Decay (Epoch 10 to E_P2=40)
        # ----------------------------------------------------
        # total_iters = 40 - 10 = 30
        scheduler2 = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=FACTOR_P2_END, # Decays from 1.0 down to LR_P2_END / LR_START
            total_iters=(E_P2 - E_P1) 
        )
        
        # ----------------------------------------------------
        # Phase 3: Long-term Fine-tuning Decay (Epoch 40 to End)
        # ----------------------------------------------------
        # total_iters = EPOCHS_TOTAL - 40 (e.g., 150 - 40 = 110)
        scheduler3 = LinearLR(
            optimizer, 
            start_factor=FACTOR_P2_END, # Starts where Phase 2 ended (at 4.0e-5 / 1.5e-4)
            end_factor=FINAL_END_FACTOR, # Ends at the overall target (1.0e-6 / 1.5e-4)
            total_iters=(EPOCHS_TOTAL - E_P2) 
        )
        
        # ----------------------------------------------------
        # Sequential Implementation
        # ----------------------------------------------------
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[E_P1, E_P2] 
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', 
                'frequency': 1,
            }
        }
import secrets
import time
def main():
   
    pl.seed_everything(42)
    optimal_config_values = {
        'TASK': 1,
        'augment': True,
        'only_mask': True,
        'use_reflection': False, 
        'batch_size': 512,
        'embedding_dim': 768,
        'embedding_dropout_rate': 0.04, 
        'epochs': 400,
        'lr': 0.00015,
        'mlp_dropout_rate': 0.1,
        'mlp_size': 128,
        'num_heads': 64,
        'num_transformer_layers': 4,
        'scheduler': True,
        'weight_decay':5e-2, 
        'grad_clip':2
    }   
    
    TASK = optimal_config_values['TASK']

    run_name = f"{secrets.token_hex(4)}"
    torch.set_float32_matmul_precision('medium')
    wandb.finish()
    wandb.init(project=f"ViT-Replication-QM9-Regression-Task{TASK}", config=optimal_config_values, name=run_name)
    config = wandb.config 

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

    if config.only_mask: 
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

    if config.use_reflection:
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


    final_train_df = pd.concat([real_train_df, train_aug1], ignore_index=True)

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

    # --- Create Datasets ---
    train_dataset = MoleculeSequenceDataset(X_train_scaled, y_train_scaled, augment=config.augment)
    val_dataset = MoleculeSequenceDataset(X_val_scaled, y_val_scaled)
    test_dataset = MoleculeSequenceDataset(X_test_scaled, y_test_scaled)

    data_module = QMDataModule(batch_size=config.batch_size) 
    data_module.set_datasets(train_dataset, val_dataset, test_dataset)

    model = ViTModule(learning_rate=config.lr, 
                        embedding_dim=config.embedding_dim, 
                        embedding_dropout_rate=config.embedding_dropout_rate, 
                        mlp_dropout_rate=config.mlp_dropout_rate,
                        num_transformer_layers=config.num_transformer_layers,
                        num_heads=config.num_heads,
                        mlp_size=config.mlp_size,
                        scaler=y_scaler,
                        weight_decay=config.weight_decay, 
                        test_ids=test_ids_to_pass, 
                        output_file_name=run_name

                        )
    wandb_logger = WandbLogger(project=f'ViT-Replication-QM9-Regression-Task{TASK}', name=run_name)

    trainer = pl.Trainer(
        max_epochs=config.epochs, 
        accelerator='auto',
        logger=wandb_logger,      
        gradient_clip_val=config.grad_clip, 
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

    end_time   = time.time()
    print(end_time - start_time)
if __name__ == "__main__":
    main()
