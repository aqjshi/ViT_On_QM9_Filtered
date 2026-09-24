import sys 
import os 
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pytorch_lightning as pl


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from core.augmentation import rotate_molecule, translate_molecule, reflect_molecule


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


    
