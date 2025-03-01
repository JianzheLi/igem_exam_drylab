import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dataset import create_loaders,DNADataset
from torch.utils.data import DataLoader
from config import config
from model import DNA_Transformer, DNA_LSTM

def preprocess_onehot(onehot_features):
   
    seq_indices = np.argmax(onehot_features, axis=2)  # (n, 70)
    return seq_indices.astype(float)

def visualize_dimension_reduction(features, labels, method='pca'):
   
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization'
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
        title = 'UMAP Visualization'
    else:
        raise ValueError("Method must be 'pca' or 'umap'")
    
    embeddings = reducer.fit_transform(features)
    

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    
    scatter = plt.scatter(
        embeddings[:, 0], 
        embeddings[:, 1],
        c=labels,
        cmap='Spectral',
        s=10,
        alpha=0.7
    )
    

    legend = plt.legend(
        *scatter.legend_elements(),
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
  
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{title} of DNA Sequences (n={len(labels)})')

    plt.savefig(f'{method}_visualization.png', bbox_inches='tight')
    plt.show()


def extract_encodings(model, dataloader, device='cuda'):

    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            #print(inputs.shape)
            #  [batch_size, seq_len, d_model]
            embeddings = model.get_encode(inputs)
            
            #[batch_size, seq_len * d_model]
            batch_size = embeddings.shape[0]
            #print(embeddings.shape)
            flattened = embeddings.reshape(batch_size, -1).cpu().numpy()
            
            all_embeddings.append(flattened)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_embeddings), np.concatenate(all_labels)

def visualize_embeddings(embeddings, labels, method='pca', save_path=None):
    
   
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization'
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=0)
        title = 'UMAP Visualization'
    else:
        raise ValueError("Method must be 'pca' or 'umap'")
    
    projected = reducer.fit_transform(embeddings)
    

    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
   
    scatter = plt.scatter(
        projected[:, 0], 
        projected[:, 1], 
        c=labels, 
        cmap='Spectral',
        s=10,
        alpha=0.7
    )
    
 
    legend = plt.legend(
        *scatter.legend_elements(),
        title="Classes",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
  
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{title} of DNA Sequence Embeddings (n={len(labels)})')
    
   
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    data=np.load('./data/dna_data.npz')
    onehot_features = data['features']  # (n, 70, 4)
    labels = data['labels']              # (n,)
    
    all_ds = DNADataset(
        features=onehot_features,
        labels=labels,
        is_train=True  
    )

    data_loader = DataLoader(
        all_ds,
        batch_size=256,
        shuffle=True,
    )
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

    embeddings, labels = extract_encodings(model, data_loader, device)

    visualize_embeddings(
        embeddings, 
        labels,
        method='pca',
        save_path=f"{config['model']}_pca.png"
    )
    
   
    visualize_embeddings(
        embeddings,
        labels,
        method='umap',
        save_path=f"{config['model']}_umap.png"
    )


    
    """processed_features = preprocess_onehot(onehot_features)
    
    
    visualize_dimension_reduction(processed_features, labels, method='pca')
    
  
    visualize_dimension_reduction(processed_features, labels, method='umap')"""


#事实证明如果只用tranformer，不能再latent space把二者区分，故不展示结果