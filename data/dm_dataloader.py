from typing import List, Dict

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json

import torch.distributions as D


def to_gauss_from_uniform(x, eps=1e-6):
    # x, a, b 形状可广播；要求 a < b
    # x in (0, 1)
    u = x.to('cuda')
    u = torch.clamp(u, eps, 1.0 - eps)
    # probit: 正态分位（icdf）
    z = D.Normal(0.0, 1.0).icdf(u)
    z = z.to(x.device)
    return z

def from_gauss_to_uniform(z):
    # return u in (0, 1)
    u = D.Normal(0.0, 1.0).cdf(z)
    return u

class ParamMeta:
    """
    Helper class to store parameter metadata for VST parameters.
    Caches sorted parameter keys, ranges, and category sizes.
    """
    
    def __init__(self, config_path):
        """
        Args:
            config_path: path to config JSON file with 'ranges' and 'num_categories'
        """
        # Load config once
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.ranges = config["ranges"]
        self.num_categories = config["num_categories"]
        
        # Sort and cache parameter keys for consistent ordering
        self.cont_param_keys = sorted(self.ranges.keys())
        self.cat_param_keys = sorted(self.num_categories.keys())
        
        # Pre-compute dimensions
        self.num_cont_params = len(self.cont_param_keys)
        self.total_cat_dims = sum(self.num_categories[k] for k in self.cat_param_keys)
        
        # Cache ranges and category counts in sorted order
        self.cont_ranges = [(self.ranges[k][0], self.ranges[k][1]) for k in self.cont_param_keys]
        self.cat_sizes = [self.num_categories[k] for k in self.cat_param_keys]

        self.non_bool_keys = set()
        self.bool_keys = set()
        for k in self.ranges.keys():
            self.non_bool_keys.add(k)
        for k in self.num_categories.keys():
            if self.num_categories[k] > 2:
                self.non_bool_keys.add(k)
            else:
                self.bool_keys.add(k)

    def param_encode(self, param_keys: List[str], params: Dict[str, float]) -> torch.Tensor:
        """
        Encode a parameter dict into continuous and categorical tensors.
        
        Args:
            param_keys: list of all parameter keys in order
            params: dict mapping parameter names to values
        Returns:
            x: a tensor embedding of shape (num_params,)
        """
        x_list = []
        for k in param_keys:
            if k in self.non_bool_keys:
                val = torch.tensor([params[k]], dtype=torch.float32)
                x_list.append(val)
            elif k in self.bool_keys:
                cat_idx = int(params[k])
                if cat_idx == 0:
                    val = torch.tensor([0.25], dtype=torch.float32)
                else:
                    val = torch.tensor([0.75], dtype=torch.float32)
                x_list.append(val)
            else:
                raise ValueError(f"Unknown parameter key: {k}")
        
        x_all = torch.cat(x_list, dim=0)
        
        return x_all
    
    def param_decode(self, param_keys: List[str], x_all) -> Dict[str, float]:
        """
        Decode continuous and categorical tensors back into a parameter dict.
        
        Args:
            param_keys: list of all parameter keys in order
            x_all: a tensor of shape (num_params,)
        Returns:
            params: dict mapping parameter names to values
        """
        params = {}
        for i, k in enumerate(param_keys):
            val = x_all[i]
            if k in self.non_bool_keys:
                u = from_gauss_to_uniform(val).item()
                params[k] = u
            elif k in self.bool_keys:
                u = val.item() # in (-inf, +inf)
                if u < 0:
                    cat_idx = 0
                else:
                    cat_idx = 1
                params[k] = cat_idx
            else:
                raise ValueError(f"Unknown parameter key: {k}")
        
        return params
            

class ParamAudioDataset(Dataset):
    """
    Dataset that contains VST parameter tensors and associated audio features embeddings.
    Loads config once and caches sorted parameter keys for efficiency.
    """
    def __init__(self, config_path, data_folder, device="cpu", dtype=torch.bfloat16):
        """
        Args:
            config_path: path to config JSON file with 'ranges' and 'num_categories'
            data_folder: path to the folder containing data chunks
            device: torch device to place tensors on
            dtype: data type for tensors
        """
        super().__init__()
        print("Initializing ParamAudioDataset...")
        self.device = device
        self.dtype = dtype
        
        # Load parameter metadata
        self.param_meta = ParamMeta(config_path)
        
        # Check the number of chunks in the data folder
        number_chunks = len([name for name in os.listdir(data_folder) if name.startswith("chunk") and name.endswith(".pkl")])
        print(f"Found {number_chunks} data chunks in {data_folder}.")

        # Read all data chunks and concatenate
        prompt_emedding_list = []
        param_list = []
        for i in range(number_chunks):
            file_path = os.path.join(data_folder, f"chunk_{i:06d}.pkl")
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                for d in data:
                    prompt_emedding_list.append(d['features'])
                    param_list.append(d['params'])
        
        self.embeddings = torch.stack(prompt_emedding_list).to(device=self.device, dtype=self.dtype)
        self.num_samples = len(self.embeddings)
        print(f"Total samples loaded: {self.num_samples}")

        param_keys = param_list[0].keys()
        self.sorted_param_keys = []
        for k in self.param_meta.cont_param_keys:
            if k in param_keys:
                self.sorted_param_keys.append(k)
        for k in self.param_meta.cat_param_keys:
            if k in param_keys:
                self.sorted_param_keys.append(k)
        print(f"Using {len(self.sorted_param_keys)} parameter keys for encoding.")
        
        param_embedding_list = []
        for params in tqdm(param_list, desc="Encoding parameters"):
            param_tensor = self.param_meta.param_encode(self.sorted_param_keys, params)
            param_embedding_list.append(param_tensor)
        print(f"Parameter tensors encoded.")        
        
        param_tensors = to_gauss_from_uniform(torch.stack(param_embedding_list).to(device=self.device, dtype=self.dtype))
        

        # Calculate Mean and Std for each parameter
        print("Calculating parameter means and stds...")
        self.means = torch.mean(param_tensors, dim=0, keepdim=True)  # (1, num_params)
        self.stds = torch.std(param_tensors, dim=0, keepdim=True) + 1e-4    # (1, num_params)
        
        # Standardize parameters
        self.param_tensors = (param_tensors - self.means) / self.stds
        self.param_tensors = self.param_tensors.to(device=self.device, dtype=self.dtype)

        print(f"Initialization complete.")

    def from_vector_to_params(self, x_all: torch.Tensor) -> Dict[str, float]:
        """
        Decode a batch of parameter tensors back into parameter dicts.
        
        Args:
            x_all: tensor of shape (1, num_params)
        Returns:
            List of parameter dicts
        """
        # De-standardize
        x_all = x_all * self.stds + self.means  # (1, num_params)
        
        params = self.param_meta.param_decode(self.sorted_param_keys, x_all.squeeze(0))
        return params

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a single sample containing audio features and parameter tensor.
        
        Returns:
            dict with keys:
                - 'features': tensor of shape (feature_dim,)
                - 'params': tensor of shape (num_cont_params + total_cat_dims,)
        """
        sample = {
            'features': self.embeddings[idx],
            'params': self.param_tensors[idx]
        }
        return sample
    

if __name__ == "__main__":
    dataset = ParamAudioDataset(
        config_path="config.json",
        data_folder="../my_data/",
        device="cpu",
        dtype=torch.float32
    )

    print(dataset.param_meta.bool_keys)

    # Get a single sample
    for i in range(5):
        sample = dataset[i]
        print(f"\nSingle sample:")
        print(f"  Features shape: {sample['features'].shape}, dtype: {sample['features'].dtype}")
        print(f"  Params shape: {sample['params'].shape}, dtype: {sample['params'].dtype}")

    cache_path = "cache/param_audio_dataset.pt"
    torch.save(dataset, cache_path)
    print(f"\nDataset saved to {cache_path}")
    
    # We have 120080 samples in total.
    # We randomly pick 100000 samples for training, and the rest for validation.
    # train_size = 100000
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
# 
    # # save dataset to cache
    # train_data_path = "cache/train_dataset.pt"
    # val_data_path = "cache/val_dataset.pt"
    # 
    # torch.save(train_dataset, train_data_path)
    # torch.save(val_dataset, val_data_path)
    # print(f"Dataset saved to {train_data_path} and {val_data_path}")
