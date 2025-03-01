import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,random_split

class DNADataset(Dataset):
    def __init__(self, features, labels, is_train=False):
        self.features = features
        self.labels = labels
        self.is_train = is_train

        assert len(self.features) == len(self.labels), "Features and labels must have the same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.features[idx].copy()).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.is_train:
            seq = self._augment_sequence(seq)

        return seq, label

    def _augment_sequence(self, seq):
        # 反向互补（50%概率）
        if torch.rand(1) < 0.5:
            seq = torch.flip(seq, dims=[0])  # 反向
            seq = seq[:, [1, 0, 3, 2]]      # 互补（交换AT和CG）
        return seq

def create_datasets(npz_path):
  
    data = np.load(npz_path)
    features = data['features']
    labels = data['labels']
    
    generator = torch.Generator().manual_seed(0)
    total = len(labels)
    indices = torch.randperm(total, generator=generator).tolist()

  
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
  
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

  
    train_ds = DNADataset(
        features=features[train_idx],
        labels=labels[train_idx],
        is_train=True  
    )
 
    val_ds = DNADataset(features[val_idx], labels[val_idx])
    test_ds = DNADataset(features[test_idx], labels[test_idx])

    return train_ds, val_ds, test_ds

def create_loaders(npz_path, batch_size=128):
    train_ds, val_ds, test_ds = create_datasets(npz_path)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_loaders('./data/dna_data.npz')


    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")


    train_features, _ = next(iter(train_loader))
    val_features, _ = next(iter(val_loader))
    
    print("\ntrain sample:")
    print("feature:", train_features[0].shape)
