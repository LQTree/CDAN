import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import warnings
warnings.filterwarnings('ignore')

class MVSAImageTextDataset(Dataset):
    """MVSA格式数据集加载器 - 支持训练/验证/测试三模式"""
    
    def __init__(
        self,
        data_dir: str,
        label_file: str,
        mode: str = 'train',  # 'train', 'val', 'test'
        img_size: int = 224,
        max_text_len: int = 128,
        balance_classes: bool = False  # 是否启用类别平衡
    ):
        """
        Args:
            data_dir: 存放.jpg/.txt文件的目录 (e.g., '/project5/data')
            label_file: 标签CSV文件路径 (train.txt 或 test_without_label.txt)
            mode: 'train'/'val'/'test' - 测试集自动忽略标签
            img_size: 图像缩放尺寸
            max_text_len: 文本最大长度
            balance_classes: 训练时是否启用类别平衡采样
        """
        self.data_dir = data_dir
        self.mode = mode
        self.max_text_len = max_text_len
        
        # 图像预处理 (ImageNet标准化)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # 加载标签文件
        self.df = pd.read_csv(label_file)
        self.df = self.df[self.df['guid'].notna()]  # 移除空guid
        
        # 数据清洗与验证
        self._validate_and_clean()
        
        # 类别映射 (与CDAN论文一致)
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # 仅训练/验证模式需要标签
        if mode != 'test':
            # 过滤无效标签
            valid_mask = self.df['tag'].isin(self.label_map.keys())
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"⚠️ 警告: 过滤 {invalid_count} 个无效标签 (非positive/neutral/negative)")
                self.df = self.df[valid_mask].reset_index(drop=True)
            
            # 生成标签索引
            self.labels = self.df['tag'].map(self.label_map).values.astype(np.int64)
            
            # 类别分布统计
            self._print_class_distribution()
            
            # 类别平衡权重 (用于WeightedRandomSampler)
            if balance_classes and mode == 'train':
                self.class_weights = self._compute_class_weights()
            else:
                self.class_weights = None
        else:
            self.labels = None
            self.class_weights = None
    
    def _validate_and_clean(self):
        """验证文件存在性并清理无效样本"""
        valid_indices = []
        missing_files = []
        
        for idx, row in self.df.iterrows():
            guid = int(row['guid'])
            img_path = os.path.join(self.data_dir, f"{guid}.jpg")
            txt_path = os.path.join(self.data_dir, f"{guid}.txt")
            
            # 检查文件是否存在
            if not (os.path.exists(img_path) and os.path.exists(txt_path)):
                missing_files.append(guid)
                continue
            
            # 检查文本是否为空
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                if not text:
                    continue
            except Exception:
                continue
            
            valid_indices.append(idx)
        
        # 应用过滤
        original_len = len(self.df)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # 打印清理报告
        if missing_files:
            print(f"▶ 清理报告: 原始样本数={original_len}, 有效样本数={len(self.df)}")
            print(f"   - 缺失文件样本数: {len(missing_files)} (示例: {missing_files[:5]})")
        else:
            print(f"✔ 数据验证通过: 共 {len(self.df)} 个有效样本")
    
    def _print_class_distribution(self):
        """打印类别分布统计"""
        if self.labels is None:
            return
        
        total = len(self.labels)
        counts = np.bincount(self.labels, minlength=3)
        ratios = counts / total * 100
        
        print("\n▶ 类别分布统计:")
        for i, label_name in enumerate(['positive', 'neutral', 'negative']):
            print(f"   {label_name:12s}: {counts[i]:5d} 样本 ({ratios[i]:5.2f}%)")
        print(f"   {'总计':12s}: {total:5d} 样本")
        
        # 检测严重不平衡 (negative < 15%)
        if ratios[2] < 15.0:
            print(f"×  警告: negative类别占比过低({ratios[2]:.2f}%)，建议启用balance_classes=True")
    
    def _compute_class_weights(self):
        """计算类别权重用于平衡采样 (解决MVSA数据集不平衡问题)"""
        counts = np.bincount(self.labels, minlength=3)
        weights = 1.0 / (counts + 1e-5)  # 避免除零
        weights = weights / weights.sum() * len(weights)  # 归一化
        sample_weights = weights[self.labels]
        return torch.from_numpy(sample_weights).float()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        guid = int(self.df.iloc[idx]['guid'])
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        
        # 加载并预处理图像
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # 图像加载失败时返回零张量 (训练中罕见，但需鲁棒处理)
            print(f"× 图像加载失败 {img_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # 加载并清理文本
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            # 文本清洗: 移除URL/特殊字符 (MVSA常见噪声)
            import re
            text = re.sub(r'http\S+|www\S+', '', text)  # 移除URL
            text = re.sub(r'@\w+|#\w+', '', text)       # 移除@提及和#标签
            text = re.sub(r'\s+', ' ', text).strip()     # 合并多余空格
            
            # 2. 长度预截断（预防CLIP截断丢失关键情感词）
            if len(text) > 200:  # 200字符 ≈ 40-50 tokens
                text = text[:200] + "..."  # 保留开头关键信息
            
            if not text:
                text = "[EMPTY]"  # 空文本占位符
        except Exception as e:
            print(f"× 文本加载失败 {txt_path}: {e}")
            text = "[ERROR]"
        
        # 返回数据结构
        sample = {
            'guid': guid,
            'image': image,
            'text': text[:self.max_text_len]  # 截断长文本
        }
        
        # 仅训练/验证模式返回标签
        if self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return sample


def create_dataloaders(
    data_dir: str,
    train_label_file: str,
    test_label_file: str,
    batch_size: int = 16,
    val_split: float = 0.15,  # 验证集比例
    balance_classes: bool = True,  # 启用类别平衡采样
    num_workers: int = 4,
    seed: int = 42
):
    """
    创建训练/验证/测试DataLoader (自动划分验证集)
    
    Returns:
        train_loader, val_loader, test_loader, class_weights (用于损失函数)
    """
    # 固定随机种子保证可复现性
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. 创建完整训练集 (含标签)
    full_train_dataset = MVSAImageTextDataset(
        data_dir=data_dir,
        label_file=train_label_file,
        mode='train',
        balance_classes=balance_classes
    )
    
    # 2. 按类别分层划分训练/验证集 (解决不平衡数据划分偏差)
    indices = np.arange(len(full_train_dataset))
    labels = full_train_dataset.labels
    
    train_indices, val_indices = [], []
    for class_id in np.unique(labels):
        class_indices = indices[labels == class_id]
        np.random.shuffle(class_indices)
        split_point = int(len(class_indices) * (1 - val_split))
        train_indices.extend(class_indices[:split_point])
        val_indices.extend(class_indices[split_point:])
    
    # 创建子集
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_indices)
    
    # 3. 创建测试集 (无标签)
    test_dataset = MVSAImageTextDataset(
        data_dir=data_dir,
        label_file=test_label_file,
        mode='test'
    )
    
    # 4. 构建DataLoader
    # 训练集: 启用加权采样解决类别不平衡
    if balance_classes and full_train_dataset.class_weights is not None:
        train_weights = full_train_dataset.class_weights[train_indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 验证/测试集: 顺序采样
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证可更大batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 5. 返回类别权重 (用于损失函数加权)
    # 根据MVSA分布: positive 59.5%, neutral 30.1%, negative 10.4%
    class_weights = torch.tensor([1.0, 1.8, 5.0], dtype=torch.float)  # negative权重最高
    
    print(f"\n✔ DataLoader构建完成:")
    print(f"   训练集: {len(train_dataset)} 样本 | 验证集: {len(val_dataset)} 样本 | 测试集: {len(test_dataset)} 样本")
    print(f"   Batch Size: {batch_size} | 启用类别平衡: {balance_classes}")
    
    return train_loader, val_loader, test_loader, class_weights

def create_fixed_val_loader(data_dir, val_guid_file, batch_size=32, num_workers=4):
    """
    创建固定验证集DataLoader（不重新划分）
    """
    # 读取保存的GUID
    val_guids = pd.read_csv(val_guid_file)['guid'].tolist()
    
    # 创建仅包含这些GUID的Dataset
    class FixedValDataset(MVSAImageTextDataset):
        def __init__(self, data_dir, guid_list):
            self.data_dir = data_dir
            self.guid_list = guid_list
            
            # 临时创建完整df用于过滤
            temp_df = pd.read_csv(r"E:\data\project5\project5\train.txt")  # 替换为你的标签文件路径
            self.df = temp_df[temp_df['guid'].isin(guid_list)].reset_index(drop=True)
            
            # 调用父类初始化（跳过重复验证）
            self.mode = 'val'
            self.max_text_len = 128
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
            
            # 生成标签
            valid_mask = self.df['tag'].isin(self.label_map.keys())
            self.df = self.df[valid_mask].reset_index(drop=True)
            self.labels = self.df['tag'].map(self.label_map).values.astype(np.int64)
    
    dataset = FixedValDataset(data_dir, val_guids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, dataset.labels

# ==================== main ====================
if __name__ == "__main__":
    # 配置路径
    DATA_DIR = r"E:\data\project5\project5\data"
    TRAIN_LABEL = r"E:\data\project5\project5\train.txt"
    TEST_LABEL = r"E:\data\project5\project5\test_without_label.txt"
    
    # 创建DataLoader
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        data_dir=DATA_DIR,
        train_label_file=TRAIN_LABEL,
        test_label_file=TEST_LABEL,
        batch_size=16,
        val_split=0.15,
        balance_classes=True,  # 解决negative样本稀少问题
        num_workers=4
    )
    
    # 验证数据加载
    print("\n▶ 验证数据加载 (训练集首batch):")
    for batch in train_loader:
        print(f"   图像形状: {batch['image'].shape}")
        print(f"   文本示例: {batch['text'][:3]}")
        print(f"   标签分布: {np.bincount(batch['label'].numpy(), minlength=3)}")
        print(f"   GUID示例: {batch['guid'][:3].tolist()}")
        break
    
    # 验证测试集加载
    print("\n▶ 验证测试集加载:")
    for batch in test_loader:
        print(f"   测试图像形状: {batch['image'].shape}")
        print(f"   测试文本示例: {batch['text'][:3]}")
        print(f"   测试GUID: {batch['guid'][:3].tolist()}")
        break
    
    print(f"\n💡 提示: class_weights = {class_weights.tolist()} 可用于损失函数:")
    print("   criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))")