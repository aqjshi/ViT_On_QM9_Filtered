
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torchmetrics import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
import sys
import os
import secrets
import time
import secrets
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.ViT import ViT
from core.utils import npy_preprocessor, scale_x_coordinates
from core.dataset import MoleculeSequenceDataset, QMDataModule
from core.augmentation import reflect_molecule

OUT_DIR = os.path.join(PROJECT_ROOT, "out")




class ViTModule(pl.LightningModule):
    def __init__(self, in_channels, patch_size, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, decay_start_epoch, embedding_dropout, mlp_dropout, scaler, 
                 weight_decay, test_ids, output_file_name): 
        super().__init__()
        self.save_hyperparameters()
        self.test_ids =test_ids
        self.output_file_name =output_file_name
        self.model = ViT(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim, 
                         num_classes=1, 
                         embedding_dropout=embedding_dropout, 
                         mlp_dropout=mlp_dropout,
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


        bin_pred = (unscaled_preds > 0).astype(int)
        bin_true = (unscaled_labels > 0).astype(int)
        val_acc = (bin_pred == bin_true).mean()
        self.log('val/acc', float(val_acc), on_epoch=True, prog_bar=True)



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

        bin_test_pred = (unscaled_preds > 0).astype(int)
        bin_test_true = (unscaled_labels > 0).astype(int)
        test_acc = (bin_test_pred == bin_test_true).mean()
        self.log('test/acc', float(test_acc), on_epoch=True, prog_bar=True)


        results_df = pd.DataFrame({
            'item_id': self.test_ids,
            'true_value_unscaled': unscaled_labels,
            'prediction_unscaled': unscaled_preds, 
            'Unscaled Error': np.abs(unscaled_preds - unscaled_labels),
            'true_value_scaled': all_scaled_labels_flat, 
            'prediction_scaled': all_scaled_preds_flat, 
        })
        os.makedirs(OUT_DIR, exist_ok=True)
                # Use the wandb run name to make the file unique
        csv_filename = f"{self.output_file_name}.csv"
        output_file = os.path.join(OUT_DIR, csv_filename)


        results_df.to_csv(output_file, index=False)
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
            
            EPOCHS = self.trainer.max_epochs
          
            decay_start_epoch = self.hparams.decay_start_epoch
        
       
            optimizer = torch.optim.AdamW( 
                params=self.parameters(), 
                lr=self.hparams.learning_rate, 
                weight_decay=self.hparams.weight_decay,  
                eps=1e-7, 
                betas=(0.8, 0.99)
            )
    
    
            scheduler_initial = LinearLR(
                optimizer, 
                start_factor=0.5, 
                end_factor=1.0, 
                total_iters=decay_start_epoch
            )
        
            scheduler_decay = LinearLR(
                optimizer, 
                start_factor=1.0, 
                end_factor=0.1,
                total_iters=(EPOCHS - decay_start_epoch)
            )
    
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_initial, scheduler_decay],
                milestones=[decay_start_epoch]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch', 
                    'frequency': 1,
                }
            }

def main():

    policy_path = sys.argv[1] 
    optimal_config_values = json.load(open(policy_path))
    pl.seed_everything(optimal_config_values['seed'])
    TASK = optimal_config_values['TASK']
    run_name = f"{secrets.token_hex(4)}"
    torch.set_float32_matmul_precision('medium')
    wandb.finish()
    wandb.init(project=f"ViT-QM9-Regression-{TASK}", config=optimal_config_values, name=run_name)
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

    raw_combined_train_df = pd.concat([real_train_df, train_aug1], ignore_index=True)

    y_train_raw = np.stack(raw_combined_train_df['rotation'].values)[:, 1].astype(float)
    
    mean_orig = np.mean(y_train_raw)
    std_orig = np.std(y_train_raw)
    z_scores_orig = (y_train_raw - mean_orig) / std_orig

    # ----------------------------------------------------
    # 3. Create Inlier Mask & Filter DataFrame
    # ----------------------------------------------------
    z_thresh = optimal_config_values.get('train_omit_z_score', 3.0)
    
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

    # --- Create Datasets ---
    train_dataset = MoleculeSequenceDataset(X_train_scaled, y_train_scaled, augment=config.augment)
    val_dataset = MoleculeSequenceDataset(X_val_scaled, y_val_scaled)
    test_dataset = MoleculeSequenceDataset(X_test_scaled, y_test_scaled)

    data_module = QMDataModule(batch_size=config.batch_size) 
    data_module.set_datasets(train_dataset, val_dataset, test_dataset)

    model = ViTModule(in_channels= config.in_channels, patch_size=config.patch_size,
                      learning_rate=config.learning_rate, 
                        embedding_dim=config.embedding_dim, 
                        embedding_dropout=config.embedding_dropout, 
                        mlp_dropout=config.mlp_dropout,
                        num_transformer_layers=config.num_transformer_layers,
                        num_heads=config.num_heads,
                        mlp_size=config.mlp_size,
                        scaler=y_scaler,
                        decay_start_epoch=config.decay_start_epoch,
                        weight_decay=config.weight_decay, 
                        test_ids=test_ids_to_pass, 
                        output_file_name=run_name

                        )

    
    early_stop_callback = EarlyStopping(
        monitor='val/acc', 
        min_delta=0.00, 
        patience=optimal_config_values["patience"], 
        verbose=False,
        mode='max' 
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"vit-regression-{run_name}-{{epoch:02d}}-{{val/acc:.4f}}",
        monitor="val/mae",
        mode="min",
        save_top_k=1,            # Keeps only the single best model
        save_weights_only=False, # Set to True if disk space is limited
        auto_insert_metric_name=False
    )


    wandb_logger = WandbLogger(project=f'ViT-QM9-Regression-{TASK}', name=run_name)

    trainer = pl.Trainer(
        max_epochs=config.epochs, 
        accelerator='auto',
        logger=wandb_logger,      
        gradient_clip_val=config.grad_clip, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            early_stop_callback, 
            checkpoint_callback
        ]
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    wandb.finish()

    end_time   = time.time()
    print(end_time - start_time)
if __name__ == "__main__":
    main()
