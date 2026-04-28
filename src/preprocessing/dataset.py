import torch
import h5py
from torch.utils.data import Dataset
from settings import DATA_FOLDER
import os
import numpy as np

class H5Dataset(Dataset):
    def __init__(self, filepath, max_size=None):
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.input_keys = list(f['input'].attrs['key_order'])
            self.output_keys = list(f['output'].attrs['key_order'])
            
            total_len = len(f['input'][self.input_keys[0]])
            self.dataset_len = total_len if max_size is None else min(total_len, max_size)

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.input_datasets = {k: self.file[f'input/{k}'] for k in self.input_keys}
        self.output_datasets = {k: self.file[f'output/{k}'] for k in self.output_keys}
        self.cost_dataset = self.file['C']
        
    def _to_tensor(self, val):
        """Helper para convertir datos a tensores de forma eficiente"""
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return torch.tensor(val)

    def __getitem__(self, idx):
        if self.file is None: 
            self._open_file()
            
        inputs = [self._to_tensor(self.input_datasets[k][idx]) for k in self.input_keys]
        outputs = [self._to_tensor(self.output_datasets[k][idx]) for k in self.output_keys]
        return tuple(inputs), tuple(outputs)
    
    def __len__(self):
        return self.dataset_len

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['input_datasets'] = None
        state['output_datasets'] = None
        state['cost_dataset'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None

def load_dataset(filepath, max_size=None, verbose=True):
    dataset = H5Dataset(DATA_FOLDER / filepath, max_size)
    if verbose:
        print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def load_data_from_path(filepath):
    with h5py.File(filepath, "r") as f:
        keys = list(f.attrs['key_order'])
        data = {k: f[k][:] for k in keys}
        data['C'] = f['C'][:]
        return data
    
def load_data(filename):
    return load_data_from_path(DATA_FOLDER / filename)

def generate_dataset(data_files, output_name, min_cost, max_cost, max_size):
    output_path = DATA_FOLDER / output_name
    all_data = {}
    
    for data_file in data_files:
        path = str(DATA_FOLDER / data_file)
        if os.path.exists(path):
            data = load_data_from_path(path) # Usa el orden correcto automáticamente
            if not all_data:
                all_data = {k: [] for k in data.keys()}
            for k in data:
                all_data[k].append(data[k])

    if not all_data: return

    key_order = [k for k in all_data.keys() if k != 'C']

    with h5py.File(output_path, "w") as f:
        f.attrs['key_order'] = key_order
        combined_data = {k: np.concatenate(all_data[k], axis=0) for k in all_data}
        
        mask = (combined_data['C'] >= min_cost) & (combined_data['C'] <= max_cost)
        final_len = min(np.sum(mask), max_size)
        
        for k in combined_data:
            f.create_dataset(k, data=combined_data[k][mask][:max_size])

    print(f"Dataset generado exitosamente en: {output_path} (Tamaño {final_len})")