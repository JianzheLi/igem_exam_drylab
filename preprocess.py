from datasets import load_dataset
import os
import numpy as np
import torch
#os.environ["HF_DATASETS_CACHE"] = "./data"
ds = load_dataset("dnagpt/dna_core_promoter",cache_dir="./data")
#print(ds.cache_files)
#print(ds.shape)



# 变成onehot
def dna_one_hot(sequence, seq_len=70) -> torch.Tensor:
    mapping = {'A':0, 'T':1, 'C':2, 'G':3}
    
    one_hot = torch.zeros((seq_len, 4), dtype=torch.float32)
    
    for i, base in enumerate(sequence):
        
        if base in mapping:
            one_hot[i, mapping[base]] = 1.0
        else:  
            print(i)
            
    return one_hot