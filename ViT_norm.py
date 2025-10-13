import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics import Accuracy, MeanAbsoluteError
from tqdm import tqdm
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import wandb
from sklearn.metrics import accuracy_score, f1_score
import plotly.graph_objects as go
from models import npy_preprocessor
from eda import outlier, augmented_dataset, QMDataset
from ViT_yunjun import ViT, rotate_molecule, get_data
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import os
from pytorch_lightning.tuner import Tuner

class MoleculeSequenceDataset(Dataset):
    def __init__(self, X, y, scaler_mu=None, scaler_sigma=None):
        self.X = X
        self.y = y
        self.scaler_mu = scaler_mu
        self.scaler_sigma = scaler_sigma

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        molecule = self.X[idx]
        
        # 1. Convert to tensor
        x_tensor = torch.tensor(molecule, dtype=torch.float32)
        
        # 2. Apply scaling to columns 0, 1, 2 (XYZ) if scalers are provided
        if self.scaler_mu is not None and self.scaler_sigma is not None:
            # Separate continuous (XYZ) and categorical (H, C, N, O, F) features
            x_continuous = x_tensor[:, :3] 
            x_categorical = x_tensor[:, 3:]
            
            # Apply Standard Scaling: (X - mu) / sigma
            # Ensure sigma has no zero elements (add epsilon if necessary, though numpy usually handles it)
            x_scaled_continuous = (x_continuous - self.scaler_mu) / self.scaler_sigma
            
            # Recombine features
            x_tensor = torch.cat([x_scaled_continuous, x_categorical], dim=-1)

        return x_tensor, torch.tensor(self.y[idx], dtype=torch.long)
    

    
def augment_data(X_train, y_train, num_samples, rotate_molecule_func):
    X_train_stacked = np.stack(X_train)
    
    rotated_X, rotated_y = [], []
    original_num_samples = len(X_train_stacked)
    
    for _ in tqdm(range(num_samples), desc="Augmenting data"):
        idx = np.random.randint(0, original_num_samples)
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.choice(['x', 'y', 'z'])
        
        # Use the stacked array for augmentation
        aug_molecule = rotate_molecule_func(X_train_stacked[idx], angle, axis=axis)
        rotated_X.append(aug_molecule)
        rotated_y.append(y_train[idx])
        
    # Concatenate the original stacked data with the new augmented data
    X_augmented = np.concatenate((X_train_stacked, np.array(rotated_X)), axis=0)
    y_augmented = np.concatenate((y_train, np.array(rotated_y)), axis=0)
    
    print("Augmentation complete.")
    return X_augmented, y_augmented


class QMDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=64, augment=False, num_aug_samples=20_000):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.num_aug_samples = num_aug_samples
        self.scaler_mu = None
        self.scaler_sigma = None

    def setup(self, stage=None):
        # 1. Perform initial data splits
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43, stratify=self.y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=43, stratify=y_train_val
        )
        
        # 2. Fit the Scaler using the UN-AUGMENTED Training Set
        # Flatten all training XYZ coordinates into a single list
        X_train_stacked = np.stack(X_train)
        all_xyz_coords = X_train_stacked[:, :, :3].reshape(-1, 3) 
        
        # Calculate mean and standard deviation for XYZ coordinates (cols 0, 1, 2)
        # Use np.mean(axis=0) to get mean/std for each of the 3 coordinates
        self.scaler_mu = torch.tensor(np.mean(all_xyz_coords, axis=0), dtype=torch.float32)
        self.scaler_sigma = torch.tensor(np.std(all_xyz_coords, axis=0), dtype=torch.float32)
        
        # Log scaling info (helpful for debugging)
        print(f"\n--- Coordinate Scaler Fitted ---")
        print(f"XYZ Mean (Mu): {self.scaler_mu.numpy()}")
        print(f"XYZ Std Dev (Sigma): {self.scaler_sigma.numpy()}")

        # 3. Apply Augmentation (must be done AFTER fitting the scaler)
        if self.augment:
            X_train, y_train = augment_data(X_train, y_train, self.num_aug_samples, rotate_molecule)


        # 4. Create the final datasets, passing the fitted scalers
        self.train_dataset = MoleculeSequenceDataset(X_train, y_train, self.scaler_mu, self.scaler_sigma)
        self.val_dataset = MoleculeSequenceDataset(X_val, y_val, self.scaler_mu, self.scaler_sigma)
        self.test_dataset = MoleculeSequenceDataset(X_test, y_test, self.scaler_mu, self.scaler_sigma)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2, num_workers=0)

class ViTModule(pl.LightningModule):
    def __init__(self, learning_rate,embedding_dim, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViT(embedding_dim=embedding_dim, num_classes=2,             
                         embedding_dropout=embedding_dropout_rate, 
                            mlp_dropout=mlp_dropout_rate,          
                            num_transformer_layers = 6, #L
                            num_heads = 6,     #table1
                            mlp_size = 1024,     #table 1
                            attn_dropout = 0,
                            )
   
        self.criterion = nn.CrossEntropyLoss()
    
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True)
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
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)

        self.log('test/loss', loss, on_epoch=True)
        self.log('test/acc', self.test_acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.learning_rate)
        return optimizer

def find_optimal_lr(model: pl.LightningModule, datamodule: pl.LightningDataModule):
    
    # We use a temporary trainer for the finder, as it doesn't need logging or callbacks.
    temp_trainer = pl.Trainer(
        accelerator='auto',
        logger=False,
        enable_checkpointing=False
    )
    
    tuner = Tuner(temp_trainer)
    
    # Run the learning rate finder
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    
    # Get the suggestion
    suggested_lr = lr_finder.suggestion()
    
    print(f"✅ Optimal learning rate found: {suggested_lr}")
    

    fig = lr_finder.plot(suggest=True)
    # fig.show()
    fig.savefig("optlr.png")
    
    return suggested_lr

def main():
    pl.seed_everything(42)
    TASK = 1

    EPOCHS = 10
    BATCH_SIZE = 512
    AUGMENT_DATA = True
    emb_dim = 384
    emb_dropout = 0
    mlp_dropout = 0    
    df = npy_preprocessor("qm9_filtered.npy")
    if TASK == 1:
        df = df[df['chiral_centers'].apply(len)==1]

    X = df['xyz'].values 

    y = (np.stack(df['rotation'].values)[:, 1] > 0).astype(int)


    data_module = QMDataModule(X, y, batch_size=BATCH_SIZE, augment=AUGMENT_DATA)
    model = ViTModule(learning_rate=1e-7, embedding_dim= emb_dim, embedding_dropout_rate=emb_dropout, mlp_dropout_rate=mlp_dropout)

    # optimal_lr = find_optimal_lr(model, data_module)
    # 4e-05 precalculated
    optimal_lr = 4e-05


    model.hparams.learning_rate = optimal_lr


    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        logger=WandbLogger(project="ViT-Replication-QM9", name=f"yujun_TASK{TASK}_lr{optimal_lr}_layer6_head6_norm"),
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

if __name__ == "__main__":
    main()

