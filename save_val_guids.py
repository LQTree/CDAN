# save_val_guids.py - 30秒保存验证集
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocess import create_dataloaders

# 配置（与train.py一致）
CONFIG = {
    'data_dir': r"E:\data\project5\project5\data",
    'train_label_file': r"E:\data\project5\project5\train.txt",
    'test_label_file': r"E:\data\project5\project5\test_without_label.txt",
    'output_dir': r"E:\data\project5\project5\output",
    'batch_size': 16,
    'val_split': 0.15,
    'balance_classes': False,  # 保存原始分布
    'num_workers': 0,          # 避免Windows pickle问题
    'seed': 1                  # 与train.py完全一致！
}

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 固定种子
set_seed(CONFIG['seed'])

# 创建dataloader（验证集在此刻确定）
print("正在创建DataLoader以确定验证集划分...")
_, val_loader, _, _ = create_dataloaders(
    data_dir=CONFIG['data_dir'],
    train_label_file=CONFIG['train_label_file'],
    test_label_file=CONFIG['test_label_file'],
    batch_size=CONFIG['batch_size'],
    val_split=CONFIG['val_split'],
    balance_classes=CONFIG['balance_classes'],
    num_workers=CONFIG['num_workers'],
    seed=CONFIG['seed']
)

# 收集GUID
val_guids = []
val_labels = []
for batch in val_loader:
    val_guids.extend(batch['guid'].numpy().tolist())
    val_labels.extend(batch['label'].numpy().tolist())

# 保存
val_df = pd.DataFrame({'guid': val_guids, 'label': val_labels})
save_path = os.path.join(CONFIG['output_dir'], 'val_guids.csv')
val_df.to_csv(save_path, index=False)

print(f"\n✅ 验证集GUID已保存至: {save_path}")
print(f"   样本数: {len(val_guids)}")
print(f"   前5个GUID: {val_guids[:5]}")

# 打印分布
from collections import Counter
label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
dist = Counter(val_labels)
print("\n▶ 验证集类别分布:")
for label_id, count in dist.items():
    ratio = count / len(val_labels) * 100
    print(f"   {label_map[label_id]:12s}: {count:4d} 样本 ({ratio:5.2f}%)")