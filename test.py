import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from model import DNA_Transformer,DNA_LSTM
from dataset import create_loaders,DNADataset
from torch.utils.data import DataLoader
from config import config



def set_seed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_dataloader(path):
    data = np.load(path)
    features = data['features']
    labels = data['labels']
    
   
  
    all_ds = DNADataset(
        features=features,
        labels=labels,
        is_train=False
    )
    test_loader = DataLoader(
        all_ds,
        batch_size=128 * 2,
        shuffle=False,
    )
    return test_loader




def test(config,test_loader):
    model.load_state_dict(torch.load(f"{config['model_dir']}/best_model.pth"))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.float().to(device)
            output = model(data).squeeze()
            probas = torch.sigmoid(output)
            
            test_preds.extend(probas.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    
    test_preds_bin = (np.array(test_preds) > 0.5).astype(int)
    test_acc = accuracy_score(test_targets, test_preds_bin)
    test_auc = roc_auc_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds_bin)
    

    print("\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1: {test_f1:.4f}")


if __name__ == "__main__":
    set_seed(0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if config['model']=='Trans':
        model = DNA_Transformer(
            seq_length=70,
            n_features=4,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

    elif config['model']=='LSTM':
        model = DNA_LSTM(
        seq_length=70,
        n_features=4,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    else:
        exit("wrong: NO model")
    test_loader = get_dataloader('./data/dna_data.npz')
    test(config,test_loader)
