
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
        
    # Concatenate the original stacked data with the new augmented data
    X_augmented = np.concatenate((X_train_stacked, np.array(rotated_X)), axis=0)
    y_augmented = np.concatenate((y_train, np.array(rotated_y)), axis=0)
    
    print("Augmentation complete.")
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 2, num_workers=0)
    
    



class ViTModule(pl.LightningModule):
    def __init__(self, learning_rate, embedding_dim, num_transformer_layers, 
                 num_heads, mlp_size, embedding_dropout_rate=0.0, mlp_dropout_rate=0.0, scaler=None, 
                 weight_decay=0.0): # <-- ADD IT HERE
        super().__init__()
        self.save_hyperparameters()


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
        # self.criterion =   nn.MSELoss()
        
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

        # 5. PRINTING (still has one small issue)
        print("\nTEST SET SAMPLE PREDICTIONS vs. TRUE VALUES (Unscaled)")
        
        sample_data = {
            'True Value (y)': unscaled_labels[:20], 
            'Prediction (y_hat)': unscaled_preds[:20], 
            'Unscaled Error': np.abs(unscaled_preds[:20] - unscaled_labels[:20]),
            'Scaled True Value (y)': all_scaled_labels_flat[:20], 
            'Scaled Prediction (y_hat)': all_scaled_preds_flat[:20], 
        }
        df_sample = pd.DataFrame(sample_data)
        
        print(df_sample.to_string(float_format="{:.4f}".format))
        print("="*80 + "\n")

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
        
        EPOCHS = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') else 30 
        decay_start_epoch = int(EPOCHS * .2)
      
   
        optimizer = torch.optim.AdamW( params=self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, eps=1e-7, 
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
        'augment': True,
        'only_mask': True,
        'use_reflection': False, 
        'batch_size': 512,
        'emb_dim': 512,
        'emb_dropout': 0.0, 
        'epochs': 10,
        'lr': 0.00015,
        'mlp_dropout': 0.0,
        'mlp_size': 256,
        'num_heads': 8,
        'num_transformer_layers': 6,
        'scheduler': True,
        'weight_decay':1e-4, 
        'grad_clip': 1.1, 
        'num_aug_samples': 1000000,
    }   
    
    TASK = optimal_config_values['TASK']
    run_name = f"weight_decay{optimal_config_values['weight_decay']}"
    wandb.init(project=f"ViT-Replication-QM9-Regression-Task{TASK}", config=optimal_config_values, name=run_name)
    config = wandb.config 

    # 1. Split your dataframes
    df_full = npy_preprocessor("qm9_filtered.npy")
    train_val_df, test_df = train_test_split(df_full, test_size=0.2, random_state=43)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=43)
    y_scale_df= ((np.stack(train_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    
    y_scaler = RobustScaler()
    y_scaler.fit(y_scale_df) 

    train_mask = train_df['chiral_centers'].apply(len) == 1
    val_mask = val_df['chiral_centers'].apply(len) == 1
    test_mask = test_df['chiral_centers'].apply(len) == 1

    if config.only_mask:
        train_df = train_df[train_mask]
        val_df = val_df[val_mask]
        test_df = test_df[test_mask]
        train_mask = train_df['chiral_centers'].apply(len) == 1 

    if config.augment:
        if config.use_reflection:
            print("Applying chiral reflection augmentation...")
            train_chiral_df = train_df[train_mask].copy() 
            X_chiral = np.stack(train_chiral_df['xyz'].values)
            y_chiral = ((np.stack(train_chiral_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
            
            X_reflected = []
            for i in tqdm(range(len(X_chiral)), desc="Applying Chiral Reflection"):
                X_reflected.append(reflect_molecule(X_chiral[i]))
            
            train_aug1 = pd.DataFrame({
                'xyz': X_reflected,
                'rotation': [np.array([0, val[0] * -1, 0]) for val in y_chiral] # Store 'y' back in 'rotation' col
            })
        else:
            train_aug1 = pd.DataFrame(columns=['xyz', 'rotation']) 
        

        train_combined = pd.concat([train_df, train_aug1], ignore_index=True)

        print("Applying general (rot/trans) augmentation...")
        X_combined = np.stack(train_combined['xyz'].values)
        y_combined = ((np.stack(train_combined['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
        
        X_aug2_list, y_aug2_list = augment_data(
            X_combined, 
            y_combined.flatten(), 
            config.num_aug_samples
        )
        
        train_aug2 = pd.DataFrame({
            'xyz': [X_aug2_list[i] for i in range(len(X_aug2_list))],
            'rotation': [np.array([0, val, 0]) for val in y_aug2_list]
        })
        # Final DF = original + reflection (if any) + general aug
        final_train_df = pd.concat([train_combined, train_aug2], ignore_index=True)
    else:
        if config.use_reflection:
            print("Applying chiral reflection augmentation...")
            train_chiral_df = train_df[train_mask].copy() 
            X_chiral = np.stack(train_chiral_df['xyz'].values)
            y_chiral = ((np.stack(train_chiral_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
            
            X_reflected = []
            for i in tqdm(range(len(X_chiral)), desc="Applying Chiral Reflection"):
                X_reflected.append(reflect_molecule(X_chiral[i]))
            
            train_aug1 = pd.DataFrame({
                'xyz': X_reflected,
                'rotation': [np.array([0, val[0] * -1, 0]) for val in y_chiral] 
            })
        else:
            train_aug1 = pd.DataFrame(columns=['xyz', 'rotation']) 
        
        train_combined = pd.concat([train_df, train_aug1], ignore_index=True)

        final_train_df = train_combined

    # --- "starving kid" logic ---
    if TASK == 1:
        print("TASK 1 active: Filtering Val/Test and recycling scraps...")
        final_val_df = val_df[val_mask]
        final_test_df = test_df[test_mask]
        
        val_scraps_df = val_df[~val_mask]
        test_scraps_df = test_df[~test_mask]
        
        all_scraps_df = pd.concat([val_scraps_df, test_scraps_df], ignore_index=True)
        
        if not all_scraps_df.empty: # Only augment if there are scraps
            X_scraps = np.stack(all_scraps_df['xyz'].values)
            y_scraps = ((np.stack(all_scraps_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
            
            print(f"Augmenting {len(all_scraps_df)} recycled scraps...")
            X_scraps_aug_list, y_scraps_aug_list = augment_data(
                X_scraps,
                y_scraps.flatten(),
                config.num_aug_samples // 2 # Augment scraps at half-rate
            )
            
            scraps_aug_df = pd.DataFrame({
                'xyz': [X_scraps_aug_list[i] for i in range(len(X_scraps_aug_list))],
                'rotation': [np.array([0, val, 0]) for val in y_scraps_aug_list]
            })

            final_train_df = pd.concat([
                final_train_df, 
                all_scraps_df, 
                scraps_aug_df
            ], ignore_index=True)
        else:
            print("No scraps to recycle.")
    else:
        final_val_df = val_df
        final_test_df = test_df

    print(f"Final Train set size: {len(final_train_df)}")
    print(f"Final Val set size: {len(final_val_df)}")
    print(f"Final Test set size: {len(final_test_df)}")

    # --- Extract X, y arrays from the FINAL dataframes ---
    X_train = list(final_train_df['xyz'].values)
    y_train = ((np.stack(final_train_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    
    X_val = list(final_val_df['xyz'].values)
    y_val = ((np.stack(final_val_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))
    
    X_test = list(final_test_df['xyz'].values)
    y_test = ((np.stack(final_test_df['rotation'].values)[:, 1]).astype(float).reshape(-1, 1))

    # --- X Scaling ---
    # Fit scaler ONLY on the original, non-aug train data
    X_train_coords_flat = np.concatenate(list(train_df['xyz'].values))[:, :3]
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

  
    y_train_scaled = y_scaler.transform(y_train).flatten()
    y_val_scaled = y_scaler.transform(y_val).flatten()
    y_test_scaled = y_scaler.transform(y_test).flatten()

    # --- Create Datasets ---
    train_dataset = MoleculeSequenceDataset(X_train_scaled, y_train_scaled)
    val_dataset = MoleculeSequenceDataset(X_val_scaled, y_val_scaled)
    test_dataset = MoleculeSequenceDataset(X_test_scaled, y_test_scaled)

    data_module = QMDataModule(batch_size=config.batch_size) 
    data_module.set_datasets(train_dataset, val_dataset, test_dataset)

    model = ViTModule(learning_rate=config.lr, 
                        embedding_dim=config.emb_dim, 
                        embedding_dropout_rate=config.emb_dropout, 
                        mlp_dropout_rate=config.mlp_dropout,
                        num_transformer_layers=config.num_transformer_layers,
                        num_heads=config.num_heads,
                        mlp_size=config.mlp_size,
                        scaler=y_scaler,
                        weight_decay=config.weight_decay
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

    
if __name__ == "__main__":
    main()


# '''
# Starving Kid Data Max Algorithm
# 1. Load and Split
# df_full = load_data("qm9_filtered.npy")
# train_df, val_df, test_df = split_data(df_full)

# # 2. Build Base Training Set
# if config.augment:
#     # Create reflected chiral data
#     train_chiral = train_df[train_df.chiral_len == 1]
#     train_aug_reflect = reflect_data(train_chiral)
    
#     # Create general rot/trans augmentations
#     train_combined = concat(train_df, train_aug_reflect)
#     train_aug_general = augment_data(train_combined, num_samples)
    
#     # Combine all training data
#     final_train_df = concat(train_df, train_aug_reflect, train_aug_general)
# else:
#     final_train_df = train_df

# # 3. Handle Task (Filter Val/Test and Recycle Scraps)
# if config.TASK == 1:
#     # Set Val/Test to be chiral-only
#     final_val_df = val_df[val_df.chiral_len == 1]
#     final_test_df = test_df[test_df.chiral_len == 1]

#     # Get non-chiral scraps from Val/Test
#     scraps = concat(val_df[val_df.chiral_len != 1], test_df[test_df.chiral_len != 1])
    
#     # Augment and add scraps to the training pile (the "Starving Kid" part)
#     scraps_aug = augment_data(scraps, num_samples / 2)
#     final_train_df = concat(final_train_df, scraps, scraps_aug)
# else:
#     final_val_df = val_df
#     final_test_df = test_df

# # 4. Scale Data
# # Fit X scaler on *original* train coords
# x_scaler = StandardScaler().fit(train_df.X_coords)
# train_X = x_scaler.transform(final_train_df.X)
# val_X   = x_scaler.transform(final_val_df.X)
# test_X  = x_scaler.transform(final_test_df.X)

# # Fit Y scaler on *final* augmented train labels
# y_scaler = RobustScaler().fit(final_train_df.y)
# train_y = y_scaler.transform(final_train_df.y)
# val_y   = y_scaler.transform(final_val_df.y)
# test_y  = y_scaler.transform(final_test_df.y)

# # 5. Train
# datamodule = create_datamodule(train_X, train_y, val_X, val_y, test_X, test_y)
# model      = create_model(y_scaler=y_scaler)
# trainer    = create_trainer()

# trainer.fit(model, datamodule)
# trainer.test(model, datamodule)
# '''
