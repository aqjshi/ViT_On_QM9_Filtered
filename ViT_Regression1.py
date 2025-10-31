
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
               num_classes: int = 1):
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
        return torch.tensor(molecule, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

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

def translate_molecule(xyz_data, magnitude=0.02):
    """Translate the first 3 columns (x, y, z) by a random magnitude along a random axis."""
    translation_vector = np.zeros(3)
    random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
    translation_vector[random_axis] = magnitude

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    translated_coords = coords_3d + translation_vector  # Apply translation

    translated_xyz_data = np.hstack((translated_coords, other_feats))  # Still (27,8)
    return translated_xyz_data

# def reflect_molecule(xyz_data):
#     """Reflect the first 3 columns (x, y, z) across a randomly chosen axis."""
#     reflection_matrix = np.eye(3)
#     random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
#     reflection_matrix[random_axis, random_axis] = -1  # Reflect across the chosen axis

#     coords_3d = xyz_data[:, :3]    # (27,3)
#     other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
#     reflected_coords = np.dot(coords_3d, reflection_matrix.T)  # Apply reflection

#     reflected_xyz_data = np.hstack((reflected_coords, other_feats))  # Still (27,8)
#     return reflected_xyz_data

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
        
    # Concatenate the original stacked data with the new augmented data
    X_augmented = np.concatenate((X_train_stacked, np.array(rotated_X)), axis=0)
    y_augmented = np.concatenate((y_train, np.array(rotated_y)), axis=0)
    
    print("Augmentation complete.")
    return X_augmented, y_augmented


class QMDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=64, augment=False, num_aug_samples=200_000, scaler=None):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.num_aug_samples = num_aug_samples
        self.scaler = scaler 
    def setup(self, stage=None):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43
        )

        # y_train_val is now the SCALED version of the training target data.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1, random_state=43
        )

        if self.augment:
            # If augmentation involves y, y_train must be unscaled before augmentation,
            X_train, y_train = augment_data(X_train, y_train, self.num_aug_samples)
        
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
    def __init__(self, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0, scaler=None, 
                 use_clamping: bool = False,
                 clamp_range: float = 20.0):
        super().__init__()
        self.save_hyperparameters()
        self.use_clamping = use_clamping # Store the flag
        self.clamp_range = clamp_range   # Store the range

        self.model = ViT(embedding_dim=embedding_dim, 
                         num_classes=1, 
                         embedding_dropout=embedding_dropout_rate, 
                         mlp_dropout=mlp_dropout_rate, 
                         
                         # Use the variables passed from the constructor
                         num_transformer_layers = num_transformer_layers, 
                         num_heads = num_heads,
                         mlp_size = mlp_size
                         )
        self.scaler = scaler
        self.criterion =  nn.HuberLoss(delta=1.0) #
        #  self.criterion =   nn.MSELoss()
        
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
        loss = self.criterion(logits.squeeze(1), y.squeeze(1)) 
        
        self.train_mae(logits.squeeze(1), y.squeeze(1))
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/scaled_mae', self.train_mae, on_step=False, on_epoch=True, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits.squeeze(1), y.squeeze(1)) 
        
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True) 
        self.validation_step_outputs.append({'preds': logits.squeeze(1), 'labels': y.squeeze(1)}) 
        return loss
    
    def on_validation_epoch_end(self):
        # Prevent errors on sanity check
        if not self.validation_step_outputs:
            return

        all_scaled_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_scaled_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()
        self.validation_step_outputs.clear() 

        if self.use_clamping:
            all_scaled_preds = np.clip(all_scaled_preds, 
                                       -self.clamp_range, 
                                        self.clamp_range) 


        scaled_mae = np.abs(all_scaled_preds - all_scaled_labels).mean()
        self.log('val/scaled_mae', scaled_mae, on_epoch=True, prog_bar=True)
        unscaled_preds = self.scaler.inverse_transform(all_scaled_preds.reshape(-1, 1)).flatten()
        unscaled_labels = self.scaler.inverse_transform(all_scaled_labels.reshape(-1, 1)).flatten()
        
        # This is now the correct unscaled MAE
        unscaled_mae = np.abs(unscaled_preds - unscaled_labels).mean()
        self.log('val/mae', unscaled_mae, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        # Prevent errors
        if not self.test_step_outputs:
            return

        all_scaled_preds_flat = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_scaled_labels_flat = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()
        self.test_step_outputs.clear() 

        if self.use_clamping:
             all_scaled_preds_flat = np.clip(all_scaled_preds_flat, 
                                        -self.clamp_range, 
                                         self.clamp_range) 

        scaled_mae_epoch = np.abs(all_scaled_preds_flat - all_scaled_labels_flat).mean()
    
        self.log('test/scaled_mae', scaled_mae_epoch, on_epoch=True, prog_bar=True)
   

        unscaled_preds = all_scaled_preds_flat.reshape(-1, 1).flatten()
        unscaled_labels = self.scaler.transform(all_scaled_labels_flat.reshape(-1, 1)).flatten()
        

        unscaled_mae = np.abs(unscaled_preds - unscaled_labels).mean()
        self.log('test/mae', unscaled_mae, on_epoch=True, prog_bar=True)

        # This print logic will now show correct unscaled values
        print("\n" + "="*80)
        print("🧪 TEST SET SAMPLE PREDICTIONS vs. TRUE VALUES (Unscaled)")
        
        sample_data = {
            'True Value (y)': unscaled_labels[:20],       # Now correct
            'Prediction (y_hat)': unscaled_preds[:20],   # Now correct
            'Scaled Error (a.u.)': np.abs(all_scaled_preds_flat[:20] - all_scaled_labels_flat[:20]),
        }
        df_sample = pd.DataFrame(sample_data)
        
        print(df_sample.to_string(float_format="{:.4f}".format))
        print("="*80 + "\n")
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
  
        loss = self.criterion(logits.squeeze(1), y.squeeze(1)) 
        
        self.test_mae(logits.squeeze(1), y.squeeze(1)) 
        
        self.log('test/loss', loss, on_epoch=True)
        
        # *** FIX: Log as 'test/scaled_mae' for consistency ***
        self.log('test/scaled_mae', self.test_mae, on_epoch=True, prog_bar=True) 
        
        self.test_step_outputs.append({'preds': logits.squeeze(1), 'labels': y.squeeze(1)}) 
        return loss
    
        


    def configure_optimizers(self):
        
        EPOCHS = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 30 
        decay_start_epoch = int(EPOCHS * .2)
      
   
        optimizer = torch.optim.AdamW( params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-3, eps=1e-7, 
                                                    betas=(0.8, 0.99))
        scheduler_initial = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=1.0, 
            total_iters=decay_start_epoch
        )
    
        scheduler_decay = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.01,
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
    pl.seed_everything(42)
    optimal_config_values = {
        'TASK': 1,
        'augment': False,
        'batch_size': 512,
        'emb_dim': 384 ,
        'emb_dropout': 0.2, 
        'epochs': 3,
        'lr': 0.0002,
        'mlp_dropout': 0.2,
        'mlp_size': 512 ,
        'num_heads': 2,
        'num_transformer_layers': 4,
        'scheduler': True,
        'weight_decay':2.751984133978333e-05, 
        'grad_clip': 0.4677, 
        'num_aug_samples': 200000, 
        'use_clamping': False,
        'clamp_range':500.0, 
        'clip_factor': 50.0
    }

    TASK =optimal_config_values['TASK']

    wandb.init(project=f"ViT-Replication-QM9-Regression-Task{TASK}", config=optimal_config_values)
    config = wandb.config 
    run_name = f"augment={config.augment}&epochs={config.epochs}&batch_size={config.batch_size}&lr={config.lr}&scheduler={config.scheduler}&num_transformer_layers={config.num_transformer_layers}&num_heads={config.num_heads}&emb_dim={config.emb_dim}&mlp_size={config.mlp_size}&emb_dropout={config.emb_dropout}&mlp_dropout={config.mlp_dropout}"

    df = npy_preprocessor("qm9_filtered.npy")
    if TASK == 1:
        df = df[df['chiral_centers'].apply(len)==1]

    X = df['xyz'].values 
    y_original = ((np.stack(df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
        

    y_full = (y_original)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_full, test_size=0.2, random_state=43 
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=43
    )

    X_train_coords_flat = np.concatenate(X_train)[:, :3]
    
    x_coord_scaler = StandardScaler()
    x_coord_scaler.fit(X_train_coords_flat) 
    
    def scale_x_coordinates(X_split, scaler):
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
    y_stats = y_full.flatten()
    q1 = np.percentile(y_stats, 25)
    q3 = np.percentile(y_stats, 75)
    print("\n--- STATS FOR y_full (BEFORE StandardScaler) ---")
    print(f"Min:    {np.min(y_stats):.4f}")
    print(f"Max:    {np.max(y_stats):.4f}")
    print(f"Mean:   {np.mean(y_stats):.4f}")
    print(f"Std Dev:{np.std(y_stats):.4f}")
    print(f"Median: {np.median(y_stats):.4f}")
    print(f"Q1 (25%): {q1:.4f}")
    print(f"Q3 (75%): {q3:.4f}")



    q1 = np.percentile(y_stats, 25)
    q3 = np.percentile(y_stats, 75)
    iqr = q3 - q1
    
    print("\n--- STATS FOR y_train (BEFORE CLIPPING) ---")
    print(f"Q1: {q1:.4f}")
    print(f"Q3: {q3:.4f}")
    print(f"IQR: {iqr:.4f}")

    clip_factor = config.clip_factor

    clip_min = q1 - (clip_factor * iqr)
    clip_max = q3 + (clip_factor * iqr)
    
    # This print statement is now accurate

    print(f"Clipping range (Q1/Q3 +/- {clip_factor}*IQR): [{clip_min:.4f}, {clip_max:.4f}]")

    # 3. Clip ALL your datasets using these bounds
    y_train_clipped = np.clip(y_train, clip_min, clip_max)
    y_val_clipped = np.clip(y_val, clip_min, clip_max)
    y_test_clipped = np.clip(y_test, clip_min, clip_max)

    # 4. NOW, fit the RobustScaler on the *clipped* training data
    y_scaler = RobustScaler()
    y_scaler.fit(y_train_clipped)
    
    # 5. Transform all your clipped datasets
    y_train_scaled = y_scaler.transform(y_train_clipped).flatten()
    y_val_scaled = y_scaler.transform(y_val_clipped).flatten()
    y_test_scaled = y_scaler.transform(y_test_clipped).flatten()
    

    if config.augment:
        X_train_aug, y_train_scaled_aug = augment_data(X_train_scaled, y_train_scaled, config.num_aug_samples)
    else:
        X_train_aug = X_train_scaled
        y_train_scaled_aug = y_train_scaled


    data_module = QMDataModule(X, y_full, 
                               batch_size=config.batch_size, 
                               augment=config.augment,
                               scaler=y_scaler) # Pass Y scaler only

    data_module.train_dataset = MoleculeSequenceDataset(X_train_aug, y_train_scaled_aug)
    data_module.val_dataset = MoleculeSequenceDataset(X_val_scaled, y_val_scaled)
    data_module.test_dataset = MoleculeSequenceDataset(X_test_scaled, y_test_scaled)
    
    model = ViTModule(learning_rate=config.lr, 
                      embedding_dim=config.emb_dim, 
                      embedding_dropout_rate=config.emb_dropout, 
                      mlp_dropout_rate=config.mlp_dropout,
                      num_transformer_layers=config.num_transformer_layers,
                      num_heads=config.num_heads,
                      mlp_size=config.mlp_size,
                      scaler=y_scaler, 
                    use_clamping=config.use_clamping,
                      clamp_range=config.clamp_range)


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

    
if __name__ == "__main__":
    main()
