import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
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
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks.callback import Callback
import sys 
import os 
import secrets
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from core.ViT import ViT, PatchEmbedding, TransformerEncoderBlock
from core.utils import read_data, npy_preprocessor, scale_x_coordinates
from core.dataset import MoleculeSequenceDataset, QMDataModule
from core.augmentation import rotate_molecule, translate_molecule, reflect_molecule



class ViT(nn.Module):
    def __init__(self,
               in_channels: int = 8,
               patch_size: int = 1,
               num_transformer_layers: int = 8, #L
               embedding_dim: int = 216,    # Hidden size D from Table1
               num_heads: int = 8,     #table1
               mlp_size: int = 1024,     #table 1
               attn_dropout: int = 0,
               mlp_dropout: float = 0.1,
               embedding_dropout: float = 0.1,
               num_classes: int = 2):
        super().__init__()
        self.num_patches = 27
        self.class_embeddings = nn.Parameter(torch.randn(1,1,embedding_dim), requires_grad=True)

        self.position_embeddings = nn.Parameter(torch.randn(1,self.num_patches+1,embedding_dim), requires_grad=True)

        #Create the embedding dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create the patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # Create the Transformer Encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                  num_heads = num_heads,
                                                  mlp_size = mlp_size,
                                                  dropout = mlp_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim * 2),
            nn.Linear(in_features=embedding_dim * 2, out_features=embedding_dim),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embeddings.expand(batch_size, -1, -1)

        # Embed patches + explicit 3D geometry
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        # Combine [CLS] token (index 0) and mean of atom tokens (indices 1..27)
        cls_feat = x[:, 0]
        atom_mean = x[:, 1:].mean(dim=1)
        combined = torch.cat([cls_feat, atom_mean], dim=-1)

        return self.classifier(combined)


class ViTModule(pl.LightningModule):
    def __init__(self, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0, scaler=None, 
                 use_clamping: bool = False,
                 clamp_range: float = 20.0, 
                 test_ids=None,
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.test_ids =test_ids
        
        self.use_clamping = use_clamping # Store the flag
        self.clamp_range = clamp_range   # Store the range
        self.weight_decay = weight_decay   # Store the range

        self.model = ViT(embedding_dim=embedding_dim, 
                         num_classes=1, 
                         embedding_dropout=embedding_dropout_rate, 
                         mlp_dropout=mlp_dropout_rate, 

                         num_transformer_layers = num_transformer_layers, 
                         num_heads = num_heads,
                         mlp_size = mlp_size
                         )
        self.scaler = scaler
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)   # Shape: (batch_size,)
        y = y.squeeze().float()       # Shape: (batch_size,)
        
        loss = self.criterion(logits, y) 

        preds = (torch.sigmoid(logits) > 0.5).int()
        self.train_acc(preds, y.long())
        
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y.float()) 
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
        self.val_acc(preds, y)

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({'preds': preds, 'labels': y})
        return loss
    
    
    def on_validation_epoch_end(self):
   
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
        
        f1 = f1_score(all_labels, all_preds, average='weighted')
        self.log('val/f1_score', f1)
        self.validation_step_outputs.clear()
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y.float()) 
        
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
        self.test_acc(preds, y)
        self.test_step_outputs.append({'preds': preds, 'labels': y})
        
        self.log('test/loss', loss, on_epoch=True)
        self.log('test/acc', self.test_acc, on_epoch=True)
        return loss
    

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        if self.logger:
            self.logger.experiment.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    preds=all_preds,
                    y_true=all_labels,
                    class_names=['Negative', 'Positive'] 
                ),
                "global_step": self.global_step 
            })
        self.test_step_outputs.clear()


    def configure_optimizers(self):
        
        EPOCHS = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 30 
        decay_start_epoch = int(EPOCHS * .2)
      
   
        optimizer = torch.optim.AdamW( 
            params=self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,  
            eps=1e-7, 
            betas=(0.8, 0.99)
        )


        scheduler_initial = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=1.0, 
            total_iters=decay_start_epoch
        )
    
        scheduler_decay = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.70,
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

class ThresholdStopper(Callback):
    def __init__(self, monitor: str, threshold: float, check_epoch: int):
        super().__init__()
        self.monitor = monitor     # e.g., 'val/loss'
        self.threshold = threshold # e.g., 0.65
        self.check_epoch = check_epoch   # The epoch to check (1-indexed, e.g., 3)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.sanity_checking:
            return
        current_epoch_num = trainer.current_epoch + 1
        if current_epoch_num == self.check_epoch:
            current_metric = trainer.callback_metrics.get(self.monitor)
            
            if current_metric is None:
                return # Metric not available

            if current_metric > self.threshold:
                print(f"\nTriggering stop at Epoch {current_epoch_num}: "
                      f"{self.monitor} ({current_metric:.4f}) > {self.threshold}. "
                      f"Stopping training.")
                trainer.should_stop = True


def main():
    pl.seed_everything(42)


    policy_path = sys.argv[1]
    optimal_config_values = json.load(open(policy_path))
    TASK = optimal_config_values['TASK']

    wandb.init(project=f"ViT-Replication-QM9-Task{TASK}", config=optimal_config_values)
        
    config = wandb.config

    run_name = f"augment={config.augment}&epochs={config.epochs}&batch_size={config.batch_size}&lr={config.lr}&scheduler={config.scheduler}&num_transformer_layers={config.num_transformer_layers}&num_heads={config.num_heads}&emb_dim={config.emb_dim}&mlp_size={config.mlp_size}&emb_dropout={config.emb_dropout}&mlp_dropout={config.mlp_dropout}"

    df_full = npy_preprocessor("qm9_filtered.npy")
    
    df_full["binary_rotation"] = list(((np.stack(df_full["rotation"].values) > 0).astype(int)))

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
    train_chiral_df_for_aug = train_df.copy()

    if config.only_mask: 
        real_train_df = train_df
    else:
        print(f"Adding {len(df_scraps)} recycled scraps to training set...")
        real_train_df = pd.concat([train_df, df_scraps], ignore_index=False)
        print(f"Total non-augmented training set size: {len(real_train_df)}")




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
    X_val = list(final_val_df['xyz'].values)
    X_test = list(final_test_df['xyz'].values)

    y_train = (np.stack(final_train_df['binary_rotation'].values)[:, 1]).astype(float)
    y_val   = (np.stack(final_val_df['binary_rotation'].values)[:, 1]).astype(float)
    y_test  = (np.stack(final_test_df['binary_rotation'].values)[:, 1]).astype(float)

    X_train_scaled = scale_x_coordinates(X_train, x_coord_scaler)
    X_val_scaled = scale_x_coordinates(X_val, x_coord_scaler)
    X_test_scaled = scale_x_coordinates(X_test, x_coord_scaler)



    # --- Create Datasets ---
    train_dataset = MoleculeSequenceDataset(X_train_scaled, y_train, augment=config.augment)
    val_dataset = MoleculeSequenceDataset(X_val_scaled, y_val)
    test_dataset = MoleculeSequenceDataset(X_test_scaled, y_test)

    data_module = QMDataModule(batch_size=config.batch_size) 
    data_module.set_datasets(train_dataset, val_dataset, test_dataset)


    
    model = ViTModule(learning_rate=config.lr, 
                        embedding_dim=config.emb_dim, 
                        embedding_dropout_rate=config.emb_dropout, 
                        mlp_dropout_rate=config.mlp_dropout,
                        num_transformer_layers=config.num_transformer_layers,
                        num_heads=config.num_heads,
                        mlp_size=config.mlp_size,
                        scaler=None, 
                        use_clamping=config.use_clamping,
                        clamp_range=config.clamp_range,
                        weight_decay=config.weight_decay, 
                        test_ids=test_ids_to_pass
                        )

    early_stop_callback = EarlyStopping(
        monitor='val/loss', 
        min_delta=0.00, 
        patience=2, 
        verbose=False,
        mode='min' 
    )

    epoch_3_stopper = ThresholdStopper(
        monitor='val/loss_epoch',
        threshold=0.68, 
        check_epoch=9    
    )
    wandb_logger = WandbLogger(project=f'ViT-Replication-QM9-Task{TASK}', name=run_name)

    trainer = pl.Trainer(
        num_sanity_val_steps=0, 
        max_epochs=config.epochs, 
        accelerator='auto',
        logger=wandb_logger, 
        gradient_clip_val=config.grad_clip, 
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            # early_stop_callback, 
            # epoch_3_stopper  # <-- Use the new callback
        ]
    )
    
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

    
if __name__ == "__main__":

    main()

