import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os

class H5Dataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None
        
        with h5py.File(self.filepath, "r") as f:
            self.dataset_len = len(f["Y"])

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.G = self.file["G"]
        self.P = self.file["P"]
        self.I = self.file["I"]
        self.S = self.file["S"]
        self.H = self.file["H"]
        self.Y = self.file["Y"]
        
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.file is None:
            self._open_file()
            
        return (
            torch.from_numpy(self.G[idx]),
            torch.from_numpy(self.P[idx]),
            torch.from_numpy(self.I[idx]),
            self.S[idx],
            self.H[idx],
            torch.from_numpy(self.Y[idx].astype(float))
        )

def load_dataset(filepath):
    dataset = H5Dataset(DATA_FOLDER / filepath)
    print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset