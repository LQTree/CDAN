# ablation_evaluation_fixed.py - 修复Windows兼容性问题
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cdan_model import CDAN
from data_preprocess import MVSAImageTextDataset, DataLoader, transforms

# ==================== 全局定义 FixedValDataset（关键修复） ====================
class FixedValDataset(MVSAImageTextDataset):
    """可pickle的固定验证集Dataset（全局类）"""
    def __init__(self, data_dir, guid_list, label_map=None):
        self.data_dir = data_dir
        self.guid_list = guid_list
        self.max_text_len = 128
        self.mode = 'val'
        
        # 图像预处理（与训练一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # 标签映射
        self.label_map = label_map or {'positive': 0, 'neutral': 1, 'negative': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # 从标签文件加载完整df
        label_file = r"E:\data\project5\project5\train.txt"  # 替换为你的实际路径
        full_df = pd.read_csv(label_file)
        
        # 过滤出验证集样本
        self.df = full_df[full_df['guid'].isin(guid_list)].reset_index(drop=True)
        
        # 生成标签
        valid_mask = self.df['tag'].isin(self.label_map.keys())
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.labels = self.df['tag'].map(self.label_map).values.astype(np.int64)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        guid = int(self.df.iloc[idx]['guid'])
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        
        # 加载图像
        from PIL import Image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"图像加载失败 {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # 加载文本
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            import re
            text = re.sub(r'http\S+|www\S+', '', text)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 200:
                text = text[:200] + "..."
            if not text:
                text = "[EMPTY]"
        except Exception as e:
            print(f"文本加载失败 {txt_path}: {e}")
            text = "[ERROR]"
        
        return {
            'guid': guid,
            'image': image,
            'text': text[:self.max_text_len],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==================== 配置 ====================
CONFIG = {
    'model_path': r'E:\data\project5\project5\output\best_model.pth',
    'val_guid_file': r'E:\data\project5\project5\output\val_guids.csv',
    'data_dir': r'E:\data\project5\project5\data',
    'clip_path': r'E:\data\project5\project5\clip',
    'bert_path': r'E:\data\project5\project5\bert',
    'vit_path': r'E:\data\project5\project5\vit',
    'batch_size': 32,
    'num_workers': 0  # Windows必须设为0避免pickle问题
}

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model(config):
    """加载已训练CDAN模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] 使用 {device}")
    
    model = CDAN(
        num_classes=3,
        feat_dim=512,
        cmsa_layers=8,
        cmsa_heads=8,
        config=config
    ).to(device)
    
    # 加载权重
    try:
        checkpoint = torch.load(config['model_path'], map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(config['model_path'], map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"[模型] 加载成功 | 最佳验证ACC: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    return model, device

def create_fixed_val_loader(data_dir, val_guid_file, batch_size=32, num_workers=0):
    """创建固定验证集DataLoader"""
    # 读取GUID
    val_df = pd.read_csv(val_guid_file)
    guid_list = val_df['guid'].tolist()
    label_list = val_df['label'].tolist() if 'label' in val_df.columns else None
    
    # 创建Dataset
    dataset = FixedValDataset(data_dir, guid_list)
    
    # 创建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Windows必须=0
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"[数据] 固定验证集加载完成 | 样本数: {len(dataset)}")
    return loader, label_list

@torch.no_grad()
def evaluate_full(model, device, val_loader, val_labels):
    """完整CDAN模型评估"""
    print("\n[评估] CDAN Full Model")
    preds = []
    pbar = tqdm(val_loader, desc="Full Model", ncols=80)
    
    for batch in pbar:
        images = batch['image'].to(device)
        texts = batch['text']
        outputs = model(images, texts)
        preds.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
    
    acc = accuracy_score(val_labels, preds) * 100
    f1 = f1_score(val_labels, preds, average='weighted') * 100
    
    print(f"\n✅ CDAN Full 评估结果:")
    print(f"   Accuracy: {acc:.2f}%")
    print(f"   Weighted F1: {f1:.2f}%")
    print("\n📊 分类报告:")
    print(classification_report(val_labels, preds, 
                              target_names=['positive', 'neutral', 'negative']))
    return acc, f1

def main():
    set_seed(1)  # 与train.py一致
    
    print("="*60)
    print("CDAN 消融实验（使用固定验证集）")
    print("="*60)
    
    # 1. 加载模型
    model, device = load_model(CONFIG)
    
    # 2. 加载固定验证集
    val_loader, val_labels = create_fixed_val_loader(
        CONFIG['data_dir'],
        CONFIG['val_guid_file'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']  # Windows=0
    )
    
    # 3. 评估完整模型
    acc_full, f1_full = evaluate_full(model, device, val_loader, val_labels)
    
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    print(f"{'模式':<20} | {'Accuracy':<12} | {'Weighted F1':<12}")
    print("-"*60)
    print(f"{'CDAN Full':<20} | {acc_full:>10.2f}% | {f1_full:>12.2f}%")
    print("="*60)
    
    # 保存结果
    result = pd.DataFrame([{
        'mode': 'CDAN Full',
        'accuracy': acc_full,
        'f1': f1_full
    }])
    result.to_csv(os.path.join(os.path.dirname(CONFIG['model_path']), 'ablation_results.csv'), index=False)
    print(f"\n💾 结果已保存至: {os.path.join(os.path.dirname(CONFIG['model_path']), 'ablation_results.csv')}")

if __name__ == "__main__":
    main()