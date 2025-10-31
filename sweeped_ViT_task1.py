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

def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, xyz):
        pos_embedding = self.encoder(xyz)
        return pos_embedding


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=8,
                 patch_size: int = 1,
                 embedding_dim: int = 256):
        super().__init__()

        self.patcher = nn.Conv1d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.pos_encoder = PositionalEncoder(input_dim=3, embedding_dim=embedding_dim)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)                  # (batch, 8, 27)
        x_patched = self.patcher(x_permuted)             # (batch, embedding_dim, 27)
        x_patched = x_patched.permute(0, 2, 1)           # (batch, 27, embedding_dim)

        # Positional Encoding
        xyz = x[:, :, :3]                                # (batch, 27, 3)
        pos_encoding = self.pos_encoder(xyz)             # (batch, 27, embedding_dim)
        tokens = x_patched + pos_encoding
        return tokens

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=256, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=8, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output


class MLPblock(nn.Module):
    def __init__(self,
               embedding_dim: int = 256,
               mlp_size:int = 1024, # hidden units, in table 1 is mlp_size
               dropout: float = 0.1): # available in the hyperparameter section
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
                    nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=mlp_size,
                                out_features=embedding_dim),
                    nn.Dropout(p=dropout)
        )

    def forward(self,x):
        x=self.layer_norm(x)
        x=self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
               embedding_dim: int = 256,
               num_heads: int = 8,
               mlp_size: int = 1024,
               dropout: float = 0.1):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                      num_heads=num_heads,
                                                      attn_dropout=dropout)
        self.mlp_block = MLPblock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=dropout)

    def forward(self,x):
        x = x + self.msa_block(x)
        x = x + self.mlp_block(x)
        return x


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
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        # Get the batch size
        batch_size = x.shape[0]

        # Create the class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embeddings.expand(batch_size,-1,-1)    # '-1' means to infer the dimensions


        # Create the patch embedding
        x = self.patch_embedding(x)

        # Concatenate the class token to the patch embedding
        x = torch.cat((class_token,x),dim=1)
        # x = x + self.position_embeddings
        # Run Emebdding dropout
        x = self.embedding_dropout(x)

        # Pass position and patch embedding to transformer Encoder
        x = self.transformer_encoder(x)

        # Put 0th index logit through classifier (Equation 4)
        x = self.classifier(x[:,0])

        return x





class MoleculeSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        molecule = self.X[idx]
        return torch.tensor(molecule, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

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

def translate_molecule(xyz_data):
    translationx_value = np.random.normal(0, 0.002)
    translationy_value = np.random.normal(0, 0.002)
    translationz_value = np.random.normal(0, 0.002)

    translation_vector = np.array([translationx_value, translationy_value, translationz_value])
    
    coords_3d = xyz_data[:, :3]      # (27,3)
    other_feats = xyz_data[:, 3:]    # (27,5) or however many remain
    
    translated_coords = coords_3d + translation_vector 
    
    translated_xyz_data = np.hstack((translated_coords, other_feats))
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

def augment_data(X_train, y_train, num_samples):
    X_train_stacked = np.stack(X_train)
    
    rotated_X, rotated_y = [], []
    original_num_samples = len(X_train_stacked)
    
    for _ in tqdm(range(num_samples), desc="Augmenting data"):
        idx = np.random.randint(0, original_num_samples)
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.choice(['x', 'y', 'z'])
        
        # Use the stacked array for augmentation
        aug_molecule = rotate_molecule(X_train_stacked[idx], angle, axis=axis)
        aug2_molecule = translate_molecule(X_train_stacked[idx])
        rotated_X.append(aug_molecule)
        rotated_y.append(y_train[idx])
        rotated_X.append(aug2_molecule)
        rotated_y.append(y_train[idx])
    X_augmented = np.concatenate((X_train_stacked, np.array(rotated_X)), axis=0)
    y_augmented = np.concatenate((y_train, np.array(rotated_y)), axis=0)
    
    return X_augmented, y_augmented


class QMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64): 
        super().__init__()
        self.batch_size = batch_size

    def set_datasets(self, train_dataset, val_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2, num_workers=4)
    

class ViTModule(pl.LightningModule):
    def __init__(self, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0, scaler=None, 
                 use_clamping: bool = False,
                 clamp_range: float = 20.0, 
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
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
        logits = self(x)
        
        # BCE target must be float and logits must be squeezed to (batch_size,)
        loss = self.criterion(logits.squeeze(1), y.float()) 
    
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).int()
        self.train_acc(preds, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y.float()) 
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
    TASK = 1

    optimal_config_values = {
        'augment': True,
        'batch_size': 512,
        'emb_dim': 384 ,
        'emb_dropout': 0.23101135339596263 , 
        'epochs': 10,
        'lr': 0.00029,
        'mlp_dropout': 0.28261284,
        'mlp_size': 512 ,
        'num_heads': 4,
        'num_transformer_layers': 8,
        'scheduler': True,
        'weight_decay':2.751984133978333e-05, 
        'grad_clip': 0.4677, 
        'num_aug_samples': 200000, 
        'use_clamping': False,
        'clamp_range':20.0
    }
    wandb.init(project=f"ViT-Replication-QM9-Task{TASK}", config=optimal_config_values)
        
    config = wandb.config

    run_name = f"augment={config.augment}&epochs={config.epochs}&batch_size={config.batch_size}&lr={config.lr}&scheduler={config.scheduler}&num_transformer_layers={config.num_transformer_layers}&num_heads={config.num_heads}&emb_dim={config.emb_dim}&mlp_size={config.mlp_size}&emb_dropout={config.emb_dropout}&mlp_dropout={config.mlp_dropout}"

    df_full = npy_preprocessor("qm9_filtered.npy")
    X_all = df_full['xyz'].values 
    y_all = (np.stack(df_full['rotation'].values)[:, 1] > 0).astype(int)

    if TASK == 1:

        task_mask = df_full['chiral_centers'].apply(len) == 1
        
        X_task = X_all[task_mask]
        y_task = y_all[task_mask]
    
        # X_recycled = X_all[~task_mask]
        # y_recycled = y_all[~task_mask]
        X_recycled = np.array([]) 
        y_recycled = np.array([])
        print(f"Primary task samples: {len(X_task)}, Recycled samples: {len(X_recycled)}")

    else:
        X_task = X_all
        y_task = y_all
        X_recycled = np.array([]) 
        y_recycled = np.array([])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_task, y_task, test_size=0.2, random_state=43
    )
    

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=43
    )

    X_train_coords_flat = np.concatenate(X_train)[:, :3]
    x_coord_scaler = StandardScaler()
    x_coord_scaler.fit(X_train_coords_flat) 
    
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
    X_recycled_scaled = scale_x_coordinates(X_recycled, x_coord_scaler) # Scale recycled data

    # 1. Combine chiral train data with recycled data (already scaled)
    X_train_base = X_train_scaled + X_recycled_scaled
    y_train_base = np.concatenate((y_train, y_recycled))

    # --- START CHIRAL REFLECTION AUGMENTATION ---
    X_reflected, y_reflected = [], []
    print("Starting Chiral Reflection Augmentation...")
    
    # for i in tqdm(range(len(X_train_scaled)), desc="Applying Chiral Reflection"):
    #     original_molecule = X_train_scaled[i]
    #     original_label = y_train[i]
        
    #     # 1. Apply reflection transformation
    #     # reflected_molecule = r nm  eflect_molecule(original_molecule)
        
    #     # # 2. Flip the binary label (0 -> 1, 1 -> 0)
    #     # reflected_label = 1 - original_label 
        
    #     # X_reflected.append(reflected_molecule)
    #     # y_reflected.append(reflected_label)

    #     # oversampling 
    #     X_reflected.append(original_molecule)
    #     y_reflected.append(original_label)

        
    X_reflected_stacked = np.array(X_reflected)
    y_reflected_stacked = np.array(y_reflected)
    
    X_train_combined = X_train_base + [X_reflected_stacked[i, ...] for i in range(X_reflected_stacked.shape[0])]
    y_train_combined = np.concatenate((y_train_base, y_reflected_stacked))
    
    print(f"Added {len(X_reflected)} reflected samples. Total training size before rotation: {len(X_train_combined)}")


    print(f"Final training set size: {len(X_train_combined)}")
    print(f"Final validation set size: {len(X_val_scaled)}")
    print(f"Final test set size: {len(X_test_scaled)}")

    if config.augment:

        print(f"Starting augmentation of {len(X_train_combined)} samples to add {config.num_aug_samples} new ones.")
        X_train_aug, y_train_aug = augment_data(
            X_train_combined, 
            y_train_combined,
            config.num_aug_samples
        )
    else:
        X_train_aug = X_train_combined
        y_train_aug = y_train_combined
        

    train_dataset = MoleculeSequenceDataset(X_train_aug, y_train_aug)
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
                        weight_decay=config.weight_decay 
                        )

    # early_stop_callback = EarlyStopping(
    #     monitor='val/loss', 
    #     min_delta=0.00, 
    #     patience=2, 
    #     verbose=False,
    #     mode='min' 
    # )

    # epoch_3_stopper = ThresholdStopper(
    #     monitor='val/loss_epoch',
    #     threshold=0.73, 
    #     check_epoch=9    
    # )
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

