
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



class PositionalEncoder(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, xyz):
        # xyz shape: (batch_size, num_atoms, 3)
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

        # Instantiate positional encoder here
        self.pos_encoder = PositionalEncoder(input_dim=3, embedding_dim=embedding_dim)

    def forward(self, x):
        # x: (batch, atoms=27, features=8)
        x_permuted = x.permute(0, 2, 1)                  # (batch, 8, 27)
        x_patched = self.patcher(x_permuted)             # (batch, embedding_dim, 27)
        x_patched = x_patched.permute(0, 2, 1)           # (batch, 27, embedding_dim)

        # Positional Encoding
        xyz = x[:, :, :3]                                # (batch, 27, 3)
        pos_encoding = self.pos_encoder(xyz)             # (batch, 27, embedding_dim)

        # Combine patch embedding with positional encoding
        tokens = x_patched + pos_encoding
        return tokens

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=256, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=8, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # 5. Create a forward() method to pass the data through the layers
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

        # Calculate the number of patches (height*width / patchsize^2)
        self.num_patches = 27

        #Create learnable class embedding (preprend)
        self.class_embeddings = nn.Parameter(torch.randn(1,1,embedding_dim), requires_grad=True)

        #Create the position embedding
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



def get_data(task_number):
    df = np.load('qm9_filtered.npy', allow_pickle=True)
    df_X, df_y = [], []
    ##(0,1)
    if task_number == 1:
        for line in df:
            num = len(line['chiral_centers'])
            if num > 1:
                continue
            elif num == 0:
                df_X.append(line['xyz'])
                df_y.append(num)
            elif num == 1:
                df_X.append(line['xyz'])
                df_y.append(num)
        
        X = np.array(df_X)
        y = np.array(df_y)
        return X,y
    ##(R,S)
    elif task_number == 2:
        for line in df:
            num = len(line['chiral_centers'])
            if num > 1:
                continue
            elif num == 0:
                continue
            elif num == 1:
                for position, chirality in line['chiral_centers']:
                    if chirality == 'S':
                        df_X.append(line['xyz'])
                        df_y.append(0)
                    elif chirality == 'R':
                        df_X.append(line['xyz'])
                        df_y.append(1)
        
        X = np.array(df_X)
        y = np.array(df_y)
        return X,y
    ##(+,-)
    elif task_number == 3:
        for line in df:
            num = len(line['chiral_centers'])
            if num == 1:
                df_X.append(line['xyz'])  # (27, 8): [x,y,z,H,C,N,O,F]
                df_y.append(1 if line['rotation'][1] > 0 else 0)
        
        X = np.array(df_X)
        y = np.array(df_y)
        return X,y



# #start analysis
# X,y = get_data(3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
# train_dataset = MoleculeSequenceDataset(X_train, y_train)
# test_dataset = MoleculeSequenceDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = ViT().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# criterion = nn.CrossEntropyLoss()

# train_acc_list, test_acc_list, train_loss_list, test_loss_list, all_labels, all_preds = \
#     train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs= 300)

# acc = accuracy_score(all_labels, all_preds)
# precision = precision_score(all_labels, all_preds, average='weighted')
# recall = recall_score(all_labels, all_preds, average='weighted')
# f1 = f1_score(all_labels, all_preds, average='weighted')

# print(f"\n✅ Final Test Accuracy: {acc:.4f}")
# print(f"🎯 Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# plot_accuracy_and_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list)


# # In[20]:


# X,y = get_data(3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# num_samples_per_class = 1000000  
# original_num_samples = len(X_train)

# rotated_X = []
# rotated_y = []

# for i in range(num_samples_per_class):
#     idx = np.random.randint(0, original_num_samples)
#     angle = np.random.uniform(0, 2 * np.pi)
#     axis = np.random.choice(['x', 'y', 'z'])
#     aug_molecule = rotate_molecule(X_train[idx], angle, axis=axis)
#     rotated_X.append(aug_molecule)
#     rotated_y.append(y_train[idx])

# rotated_X = np.array(rotated_X)  # shape: (num_samples_per_class, 27, 8)
# rotated_y = np.array(rotated_y)  # shape: (num_samples_per_class,)

# # Combine original + augmented
# X_train_aug = np.concatenate((X_train, rotated_X), axis=0)
# y_train_aug = np.concatenate((y_train, rotated_y), axis=0)

# train_dataset = MoleculeSequenceDataset(X_train_aug, y_train_aug)
# test_dataset = MoleculeSequenceDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = ViT().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
# criterion = nn.CrossEntropyLoss()

# train_acc_list, test_acc_list, train_loss_list, test_loss_list, all_labels, all_preds = \
#     train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs= 30)

# acc = accuracy_score(all_labels, all_preds)
# precision = precision_score(all_labels, all_preds, average='weighted')
# recall = recall_score(all_labels, all_preds, average='weighted')
# f1 = f1_score(all_labels, all_preds, average='weighted')

# print(f"\n✅ Final Test Accuracy: {acc:.4f}")
# print(f"🎯 Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# plot_accuracy_and_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

# #end analysis