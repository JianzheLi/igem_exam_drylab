import json
import matplotlib.pyplot as plt
from config import config
def load_training_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_metrics(data, save_path='training_metrics.png'):
    plt.figure(figsize=(20, 15))
    
    # 1. Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(data['history']['train_loss'], label='Train Loss')
    plt.plot(data['history']['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(data['history']['train_acc'], label='Train Accuracy')
    plt.plot(data['history']['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 3. AUC曲线
    plt.subplot(2, 2, 3)
    plt.plot(data['history']['train_auc'], label='Train AUC')
    plt.plot(data['history']['val_auc'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # 4. F1 Score曲线
    plt.subplot(2, 2, 4)
    plt.plot(data['history']['train_f1'], label='Train F1')
    plt.plot(data['history']['val_f1'], label='Validation F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"save in : {save_path}")

if __name__ == "__main__":
    
    json_path = f"{config['output_dir']}/results.json" # 修改为你的JSON文件路径
    output_image = f"{config['model']}res.png"
    
    training_data = load_training_data(json_path)
    #visualize_metrics(training_data, output_image)
    history = training_data['history']
    train_acc = history['train_auc']
    val_acc = history['val_auc']

    max_train_acc = max(train_acc)
    
    max_val_acc = max(val_acc)

    #max_test_acc=max(training_data['test_acc'])
    print(max_train_acc)
    print(max_val_acc)
    print(training_data['test_auc'])