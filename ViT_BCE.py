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
from ViT_yunjun import ViT, rotate_molecule, MoleculeSequenceDataset, get_data
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
import os
from pytorch_lightning.tuner import Tuner


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
    def __init__(self, X, y, batch_size=64, augment=False, num_aug_samples=500_000):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.num_aug_samples = num_aug_samples

    def setup(self, stage=None):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43, stratify=self.y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=43, stratify=y_train_val
        )
        

        if self.augment:
            X_train, y_train = augment_data(X_train, y_train, self.num_aug_samples, rotate_molecule)


        # Create the final datasets
        self.train_dataset = MoleculeSequenceDataset(X_train, y_train)
        self.val_dataset = MoleculeSequenceDataset(X_val, y_val)
        self.test_dataset = MoleculeSequenceDataset(X_test, y_test)


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
        self.model = ViT(embedding_dim=embedding_dim, num_classes=1,             
                         embedding_dropout=embedding_dropout_rate, 
                            mlp_dropout=mlp_dropout_rate,          
                            num_transformer_layers = 6, #L
                            num_heads = 6,     #table1
                            mlp_size = 1024,     #table 1
                            )
   
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)

        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # BCE target must be float and logits must be squeezed to (batch_size,)
        loss = self.criterion(logits.squeeze(1), y.float()) 
        
        # Predictions are now based on Sigmoid(logits) > 0.5
        # Since logits are used, apply Sigmoid implicitly or check logit sign
        # torchmetrics Accuracy handles logits/probs based on 'task' setting
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
        self.train_acc(preds, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # BCE target must be float and logits must be squeezed to (batch_size,)
        loss = self.criterion(logits.squeeze(1), y.float()) 
        
        # Predictions: Convert logits to binary prediction
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
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
        
        # BCE target must be float and logits must be squeezed to (batch_size,)
        loss = self.criterion(logits.squeeze(1), y.float()) 
        
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
        self.test_acc(preds, y)

        self.log('test/loss', loss, on_epoch=True)
        self.log('test/acc', self.test_acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        
        EPOCHS = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 30 
        
        # Define the decay start point (milestone)
        decay_start_epoch = int(EPOCHS / 2)
        num_decay_epochs = EPOCHS - decay_start_epoch
        
        # Calculate the gamma factor for a total decay to 10% (0.1) over the decay period
        # gamma = (target_factor) ** (1 / num_decay_epochs)
        target_factor = 0.1
        gamma = target_factor ** (1 / num_decay_epochs) # e.g., if num_decay_epochs=15, gamma = 0.1^(1/15)
      
        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.hparams.learning_rate
        )
        
        # --- Stage 1: Constant LR (1.0 to 1.0) ---
        scheduler_initial = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=1.0, 
            total_iters=decay_start_epoch
        )
        
        # --- Stage 2: Exponential Decay ---
        # The LR multiplier will be (current_LR) * gamma every epoch.
        # This scheduler is defined relative to the LR it receives when activated.
        scheduler_decay = ExponentialLR(
            optimizer, 
            gamma=gamma 
        )

        # --- Combine Schedulers ---
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_initial, scheduler_decay],
            milestones=[decay_start_epoch]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch', # Apply the schedule step after every epoch
                'frequency': 1,
            }
        }

def find_optimal_lr(model: pl.LightningModule, datamodule: pl.LightningDataModule):
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
    
    print(f"Optimal learning rate found: {suggested_lr}")
    

    fig = lr_finder.plot(suggest=True)
    # fig.show()
    fig.savefig("optlr.png")
    
    return suggested_lr

def main():
    pl.seed_everything(42)
    TASK = 1

    EPOCHS = 25
    BATCH_SIZE = 512
    AUGMENT_DATA = True
    emb_dim = 384
    emb_dropout = 0.05
    mlp_dropout = 0.05 
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
        logger=WandbLogger(project="ViT-Replication-QM9", name=f"yujun_TASK{TASK}_aug500k_bce"),
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()

if __name__ == "__main__":
    main()

