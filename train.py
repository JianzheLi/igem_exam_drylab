from model import DNA_Transformer, DNA_LSTM
from dataset import create_loaders
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os
import json
from tqdm import tqdm
from config import config
def set_seed(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

def train_model(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    train_loader, val_loader, test_loader = create_loaders(config['data_path'], config['batch_size'])
    
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
    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    
    best_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_auc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': []
    }

    epoches = config['epochs']
    
    for epoch in range(epoches):
        train_preds, train_targets = [], []
        model.train()
        epoch_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for data, target in train_bar:
            data, target = data.to(device), target.float().to(device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            probas = torch.sigmoid(output.detach())
            train_preds.extend(probas.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_preds_bin = (train_preds > 0.5).astype(int)
        
        train_acc = accuracy_score(train_targets, train_preds_bin)
        train_auc = roc_auc_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds_bin)
        
        
        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['train_f1'].append(train_f1)

        
        
        model.eval()
        val_preds, val_targets = [], []
        epoch_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.float().to(device)
                output = model(data).squeeze()
                
                
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                
                
                probas = torch.sigmoid(output)
                val_preds.extend(probas.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_preds_bin = (val_preds > 0.5).astype(int)
        
        val_acc = accuracy_score(val_targets, val_preds_bin)
        val_auc = roc_auc_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds_bin)
        

        
        history['val_loss'].append(epoch_val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        # save
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{config['model_dir']}/best_model.pth")
            print(f"New best model saved with AUC: {best_acc:.4f}")
        
    
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Best ACC: {best_acc:.4f}")
        """print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}\n")"""
    
    # 
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
    
   
    results = {
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'history': history
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(f"{config['output_dir']}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFinal Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"F1: {test_f1:.4f}")



os.makedirs(config['model_dir'], exist_ok=True)


train_model(config)