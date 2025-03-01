config = {
    'data_path': './data/dna_data.npz',
    'batch_size': 128,
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'lr': 5e-5,
    'weight_decay': 1e-2,
    'epochs': 100,
    'model_dir': 'saved_model_LSTM',
    'output_dir': 'results_LSTM',
    'dropout': 0.3,
    'model':'LSTM'
}