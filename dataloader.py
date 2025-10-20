import torch
from torch.utils.data import Dataset, DataLoader
import json


class RandomParamDataset(Dataset):
    """
    Stateful dataset for generating random VST parameter tensors.
    Loads config once and caches sorted parameter keys for efficiency.
    """
    
    def __init__(self, config_path, num_samples, device="cpu", dtype=torch.bfloat16):
        """
        Args:
            config_path: path to config JSON file with 'ranges' and 'num_categories'
            num_samples: number of samples in an epoch
            device: torch device to place tensors on
            dtype: data type for tensors
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_samples = num_samples
        
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
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a single random parameter sample.
        
        Returns:
            dict with keys:
                - 'continuous': tensor of shape (num_cont_params,)
                - 'categorical': tensor of shape (total_cat_dims,) - one-hot encoded
        """
        # Generate continuous parameters
        continuous = []
        for min_val, max_val in self.cont_ranges:
            val = torch.rand(1, device=self.device, dtype=self.dtype) * (max_val - min_val) + min_val
            continuous.append(val)
        
        if continuous:
            continuous_tensor = torch.cat(continuous)
        else:
            continuous_tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        
        # Generate categorical parameters as one-hot
        categorical = []
        for num_cats in self.cat_sizes:
            idx = torch.randint(0, num_cats, (1,), device=self.device)
            onehot = torch.nn.functional.one_hot(idx, num_classes=num_cats).squeeze(0)
            categorical.append(onehot)
        
        if categorical:
            categorical_tensor = torch.cat(categorical).to(self.dtype)
        else:
            categorical_tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        
        return {
            'continuous': continuous_tensor,
            'categorical': categorical_tensor
        }
    
    def get_batch(self, batch_size):
        """
        Generate a batch of random parameters efficiently.
        
        Args:
            batch_size: number of samples to generate
            
        Returns:
            dict with keys:
                - 'continuous': tensor of shape (batch_size, num_cont_params)
                - 'categorical': tensor of shape (batch_size, total_cat_dims)
        """
        # Generate continuous parameters
        continuous_params = []
        for min_val, max_val in self.cont_ranges:
            vals = torch.rand(batch_size, device=self.device, dtype=self.dtype) * (max_val - min_val) + min_val
            continuous_params.append(vals.unsqueeze(1))
        
        if continuous_params:
            continuous_tensor = torch.cat(continuous_params, dim=1)
        else:
            continuous_tensor = torch.empty(batch_size, 0, device=self.device, dtype=self.dtype)
        
        # Generate categorical parameters as one-hot
        categorical_params = []
        for num_cats in self.cat_sizes:
            indices = torch.randint(0, num_cats, (batch_size,), device=self.device)
            onehot = torch.nn.functional.one_hot(indices, num_classes=num_cats)
            categorical_params.append(onehot)
        
        if categorical_params:
            categorical_tensor = torch.cat(categorical_params, dim=1).to(self.dtype)
        else:
            categorical_tensor = torch.empty(batch_size, 0, device=self.device, dtype=self.dtype)
        
        return {
            'continuous': continuous_tensor,
            'categorical': categorical_tensor
        }


def create_random_param_dataloader(config_path, num_samples, batch_size, device="cpu", dtype=torch.bfloat16, num_workers=0):
    """
    Convenience function to create a DataLoader for random parameter generation.
    
    Args:
        config_path: path to config JSON file
        num_samples: number of samples per epoch
        batch_size: batch size for DataLoader
        device: torch device
        dtype: data type for tensors
        num_workers: number of worker processes (0 for single process)
        
    Returns:
        DataLoader instance
    """
    dataset = RandomParamDataset(config_path, num_samples, device, dtype)
    
    # For random generation, we don't need shuffling
    # Also use pin_memory=True if using CUDA for faster transfers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda")
    )


if __name__ == "__main__":
    # Example usage
    print("Creating random parameter dataset...")
    
    # Example with direct batch generation (recommended for random data)
    dataset = RandomParamDataset(
        config_path="config.json",
        num_samples=1000,
        device="cpu",
        dtype=torch.bfloat16
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Continuous params: {dataset.num_cont_params}")
    print(f"Total categorical dims: {dataset.total_cat_dims}")
    
    # Get a single sample
    sample = dataset[0]
    print(f"\nSingle sample:")
    print(f"  Continuous shape: {sample['continuous'].shape}")
    print(f"  Categorical shape: {sample['categorical'].shape}")
    
    # Get a batch (more efficient for training)
    batch = dataset.get_batch(32)
    print(f"\nBatch of 32:")
    print(f"  Continuous shape: {batch['continuous'].shape}")
    print(f"  Categorical shape: {batch['categorical'].shape}")
    
    # Using standard PyTorch DataLoader
    dataloader = create_random_param_dataloader(
        config_path="config.json",
        num_samples=1000,
        batch_size=32,
        device="cpu"
    )
    
    print(f"\nDataLoader with {len(dataloader)} batches")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"First batch:")
            print(f"  Continuous shape: {batch['continuous'].shape}")
            print(f"  Categorical shape: {batch['categorical'].shape}")
        break

