from pedalboard import Pedalboard, Reverb, load_plugin, VST3Plugin
from pedalboard.io import AudioFile
from mido import Message # not part of Pedalboard, but convenient!
import random
import re
import time
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import json

def filter_osc_params(inst):
    params = inst.parameters
    ret = {}
    for k, v in params.items():
        if k.startswith('a_') or k.startswith('b_') or k.startswith('c_') \
            or k.startswith('filter_1') or k.startswith('filter_2') \
            or k.startswith('sub_') or k.startswith('arp_'):
            ret[k] = v
    return ret

def filter_cat_params(params):
    cont_params = {}
    cat_params = {}
    for k, v in params.items():
        if len(v.valid_values) < 100:
            if 'semitones' in k or 'rate' in k or v.type == float:
                cont_params[k] = v
            else:
                cat_params[k] = v
    return cat_params, cont_params

def to_onehot(params):
    ret = {}
    for k, v in params.items():
        num_values = len(v.valid_values)
        onehot = np.zeros(num_values)
        onehot[v.valid_values.index(v.value)] = 1
        ret[k] = onehot
    return ret

def get_categories(params):
    ret = {}
    for k, v in params.items():
        ret[k] = v.valid_values
    return ret

def randomize_params(inst, params):
    for k, v in params.items():
        if None not in v.range:
            low, high, step = v.range
            value = random.uniform(low, high)
            if v.type == bool:
                value = random.choice([True, False])
            setattr(inst, k, value)
        else:
            valid_values = v.valid_values
            value = valid_values[random.randint(0, len(valid_values) - 1)]
            setattr(inst, k, value)

def get_value_range(param):
    valid_values = param.valid_values
    valid_raw_values = [param.get_raw_value_for_text(v) for v in valid_values]
    min_raw, max_raw = min(valid_raw_values), max(valid_raw_values)
    step = (max_raw - min_raw) / (len(valid_values) - 1)
    return min_raw, max_raw, step

def generate_random_param_vector(ranges, num_categories):
    param_vector = {}
    for k, v in ranges.items():
        param_vector[k] = random.uniform(v[0], v[1])
    for k, v in num_categories.items():
        param_vector[k] = random.randint(0, v - 1)
    return param_vector

def generate_random_params_tensor(ranges, num_categories, batch_size = 1, device = "cpu", dtype = torch.bfloat16):
    """
    Generate random parameter tensors for batch processing.
    
    Args:
        ranges: dict mapping parameter names to (min, max, step) tuples for continuous params
        num_categories: dict mapping parameter names to number of categories for categorical params
        batch_size: number of parameter sets to generate
        device: torch device to place tensors on
        dtype: data type for continuous tensors
    
    Returns:
        tuple: (continuous_tensor, categorical_tensor)
            - continuous_tensor: shape (batch_size, num_continuous_params)
            - categorical_tensor: shape (batch_size, total_categorical_dims) - one-hot encoded
    """
    # Generate continuous parameters
    continuous_params = []
    for k in sorted(ranges.keys()):  # sort for consistent ordering
        min_val, max_val, step = ranges[k]
        # Generate random values in range [min_val, max_val]
        random_vals = torch.rand(batch_size, device=device, dtype=dtype) * (max_val - min_val) + min_val
        continuous_params.append(random_vals.unsqueeze(1))
    
    if continuous_params:
        continuous_tensor = torch.cat(continuous_params, dim=1)
    else:
        continuous_tensor = torch.empty(batch_size, 0, device=device, dtype=dtype)
    
    # Generate categorical parameters as one-hot
    categorical_params = []
    for k in sorted(num_categories.keys()):  # sort for consistent ordering
        num_cats = num_categories[k]
        # Generate random category indices
        random_indices = torch.randint(0, num_cats, (batch_size,), device=device)
        # Convert to one-hot
        onehot = torch.nn.functional.one_hot(random_indices, num_classes=num_cats)
        categorical_params.append(onehot)
    
    if categorical_params:
        categorical_tensor = torch.cat(categorical_params, dim=1).to(dtype)
    else:
        categorical_tensor = torch.empty(batch_size, 0, device=device, dtype=dtype)
    
    return continuous_tensor, categorical_tensor

def read_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def dump_config(inst, config_file=None):
    params = inst.parameters
    params = filter_osc_params(params)
    cat_params, cont_params = filter_cat_params(params)
    ranges = {}
    for k, v in cont_params.items():
        ranges[k] = v.range
    num_categories = {}
    for k, v in cat_params.items():
        num_categories[k] = len(v.valid_values)
    
    config = {
        "ranges": ranges,
        "num_categories": num_categories,
        "cont_types": str(cont_types),
        "cat_types": str(cat_types)
    }
    if config_file is not None:
        with open(config_file, "w") as f:
            json.dump(config, f)
        print(f"Config dumped to {config_file}")
    return config