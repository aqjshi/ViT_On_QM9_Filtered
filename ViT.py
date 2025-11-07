
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


