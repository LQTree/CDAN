import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')

# ======================= 【配置】 ======================
CONFIG = {
    # --- 路径配置 ---
    'data_dir': r"E:\data\project5\project5\data",           
    'train_label_file': r"E:\data\project5\project5\train.txt",
    'test_label_file': r"E:\data\project5\project5\test_without_label.txt",
    'output_dir': r"E:\data\project5\project5\output",       # 输出目录
    
    # --- 本地模型路径 (关键!) ---
    'clip_path': r"E:\data\project5\project5\clip",      # CLIP模型
    'bert_path': r"E:\data\project5\project5\bert",          # BERT模型
    'vit_path': r"E:\data\project5\project5\vit",        # ViT模型
    
    # --- 训练配置 ---
    'mode': 'train_test',                    # 'train', 'test', 'train_test'
    'epochs': 30,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'val_split': 0.15,                       # 验证集比例
    'balance_classes': True,                 
    'patience': 10,                          
    
    # --- 系统配置 ---
    'cpu': False,                            # True=强制CPU, False=自动检测GPU
    'num_workers': 4,
    'seed': 1,
    
    # --- 调试配置 ---
    'debug_samples': 0,                      
}
# ===========================================================

# 项目模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cdan_model import CDAN
from data_preprocess import create_dataloaders, MVSAImageTextDataset

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    """CDAN训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config['cpu'] else "cpu")
        print(f"▶ [设备] 使用 {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'tensorboard'))
        
        # 初始化数据加载器
        print("\n▶ [数据] 加载数据集...")
        self.train_loader, self.val_loader, self.test_loader, self.class_weights = create_dataloaders(
            data_dir=config['data_dir'],
            train_label_file=config['train_label_file'],
            test_label_file=config['test_label_file'],
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            balance_classes=config['balance_classes'],
            num_workers=config['num_workers'],
            seed=config['seed']
        )
        
        # 调试模式: 限制样本数
        if config['debug_samples'] > 0:
            print(f"🐞 [调试] 仅使用 {config['debug_samples']} 个样本进行快速验证")
            # 替换DataLoader为小批量版本
            def create_debug_loader(loader, n_samples):
                data = []
                for i, batch in enumerate(loader):
                    if i * loader.batch_size >= n_samples:
                        break
                    data.append(batch)
                return data
            self.train_loader = create_debug_loader(self.train_loader, config['debug_samples'])
            self.val_loader = create_debug_loader(self.val_loader, max(10, config['debug_samples']//5))
            self.test_loader = create_debug_loader(self.test_loader, max(10, config['debug_samples']//5))
        
        # 初始化模型
        print("\n▶ [模型] 初始化CDAN...")
        self.model = CDAN(
            num_classes=3,
            feat_dim=512,
            cmsa_layers=8,
            cmsa_heads=8,
            config=config
        ).to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and not config['cpu']:
            print(f"ParallelGroup: 检测到 {torch.cuda.device_count()} 个GPU，启用DataParallel")
            self.model = nn.DataParallel(self.model)
        
        # 损失函数 (针对MVSA不平衡分布)
        self.criterion_ce = nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.device) if self.class_weights is not None else None
        )
        self.criterion_recon = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            eps=1e-8
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # 训练状态
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        
        # 标签映射
        self.label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    
    def train_epoch(self, epoch):
        """单轮训练"""
        self.model.train()
        total_loss, total_ce_loss, total_recon_loss = 0.0, 0.0, 0.0
        correct, total = 0, 0
        
        # 使用tqdm_notebook兼容Spyder/Jupyter
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]", 
                    leave=True, file=sys.stdout)
        
        for batch_idx, batch in enumerate(pbar):
            # 处理调试模式的batch格式
            if isinstance(batch, list):
                if batch_idx >= len(self.train_loader):
                    break
                batch = self.train_loader[batch_idx]
            
            images = batch['image'].to(self.device)
            texts = batch['text']
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images, texts, labels)
            loss = outputs['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_ce_loss += outputs['ce_loss'].item()
            total_recon_loss += outputs['recon_loss'].item()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.1f}%"
            })
            
            # TensorBoard记录
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/accuracy', 100.*correct/total, self.global_step)
            
            self.global_step += 1
        
        pbar.close()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        print(f"\n▶ 训练轮次 {epoch+1} 完成 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
        self.writer.add_scalar('epoch/train_loss', avg_loss, epoch)
        self.writer.add_scalar('epoch/train_acc', accuracy, epoch)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证集评估"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]", 
                    leave=True, file=sys.stdout)
        
        for batch in pbar:
            if isinstance(batch, list):
                if not self.val_loader:
                    break
                batch = self.val_loader[0] if not batch else batch
            
            images = batch['image'].to(self.device)
            texts = batch['text']
            labels = batch['label'].to(self.device)
            
            outputs = self.model(images, texts, labels)
            loss = outputs['loss']
            total_loss += loss.item()
            
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        pbar.close()
        
        avg_loss = total_loss / (len(self.val_loader) if not isinstance(self.val_loader, list) else max(1, len(self.val_loader)))
        accuracy = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        weighted_f1 = 100. * f1_score(all_labels, all_preds, average='weighted')
        
        # 每3轮打印详细报告
        if (epoch + 1) % 3 == 0:
            print("\n▶ 分类报告:")
            print(classification_report(all_labels, all_preds, 
                                      target_names=['positive', 'neutral', 'negative']))
        
        print(f"▶ 验证轮次 {epoch+1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | F1: {weighted_f1:.2f}%")
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/accuracy', accuracy, epoch)
        self.writer.add_scalar('val/f1', weighted_f1, epoch)
        
        return avg_loss, accuracy, weighted_f1
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print("▶ 启动CDAN训练 (Spyder IDE模式)")
        print("="*70)
        print(f"  数据目录: {self.config['data_dir']}")
        print(f"  输出目录: {self.config['output_dir']}")
        print(f"  Batch Size: {self.config['batch_size']} | Epochs: {self.config['epochs']}")
        print(f"  类别平衡: {'启用' if self.config['balance_classes'] else '禁用'}")
        if self.config['debug_samples'] > 0:
            print(f"  ▶  调试模式: 仅使用 {self.config['debug_samples']} 个样本")
        print("="*70 + "\n")
        
        training_start = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # 训练 + 验证
            self.train_epoch(epoch)
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                
                model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'best_val_f1': self.best_val_f1,
                    'config': self.config
                }
                torch.save(save_dict, model_path)
                print(f"▶ 保存最佳模型 (Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%) → {model_path}")
            else:
                self.patience_counter += 1
                print(f"▶ 早停计数: {self.patience_counter}/{self.config['patience']}")
            
            # 早停
            if self.config['patience'] > 0 and self.patience_counter >= self.config['patience']:
                print(f"\n▶ 触发早停 (无改进 {self.patience_counter} 轮)")
                break
            
            epoch_time = time.time() - epoch_start
            print(f"▶  轮次耗时: {epoch_time:.2f}秒 | 学习率: {self.optimizer.param_groups[0]['lr']:.2e}\n")
        
        training_time = time.time() - training_start
        print("\n" + "="*70)
        print(f"▶ 训练完成 | 总耗时: {training_time/60:.2f} 分钟")
        print(f"   最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"   最佳验证F1: {self.best_val_f1:.2f}%")
        print("="*70)
        self.writer.close()
    
    @torch.no_grad()
    def test(self):
        """测试集预测"""
        print("\n▶ 启动测试预测...")
        
        # 加载最佳模型
        model_path = os.path.join(self.config['output_dir'], 'best_model.pth')
        if os.path.exists(model_path):
            print(f"▶ 加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            # 处理DataParallel
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
        else:
            print(f"⚠️  未找到最佳模型，使用当前权重")
        
        self.model.eval()
        results = []
        
        pbar = tqdm(self.test_loader, desc="测试预测", leave=True, file=sys.stdout)
        for batch in pbar:
            if isinstance(batch, list):
                if not self.test_loader:
                    break
                batch = self.test_loader[0] if not batch else batch
            
            images = batch['image'].to(self.device)
            texts = batch['text']
            guids = batch['guid'].numpy()
            
            outputs = self.model(images, texts)
            preds = torch.argmax(outputs['logits'], dim=1).cpu().numpy()
            
            for guid, pred in zip(guids, preds):
                results.append({'guid': int(guid), 'tag': self.label_map[pred]})
        
        pbar.close()
        
        # 保存结果
        results_df = pd.DataFrame(results).sort_values('guid').reset_index(drop=True)
        submission_path = os.path.join(self.config['output_dir'], 'submission.csv')
        results_df.to_csv(submission_path, index=False)
        
        print(f"\n▶ 预测完成 | 样本数: {len(results_df)}")
        print(f"   结果保存至: {submission_path}")
        print(f"   标签分布: {results_df['tag'].value_counts().to_dict()}")
        
        # 显示前10条预测
        print("\n▶ 前10条预测示例:")
        print(results_df.head(10).to_string(index=False))
        
        return results_df
    
def save_validation_guids(config):
    """仅创建dataloader，不训练，直接保存验证集GUID"""
    print("\n" + "="*60)
    print("▶ 保存验证集GUID（无需训练）")
    print("="*60)
    
    # 固定种子（与训练一致）
    set_seed(config['seed'])
    
    # 创建dataloader（验证集在此刻确定）
    _, val_loader, _, _ = create_dataloaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label_file'],
        test_label_file=config['test_label_file'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        balance_classes=False,  # 保存原始分布，不需过采样
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    
    # 收集所有验证集GUID
    val_guids = []
    for batch in val_loader:
        val_guids.extend(batch['guid'].numpy().tolist())
    
    # 保存为CSV
    val_df = pd.DataFrame({'guid': val_guids})
    save_path = os.path.join(config['output_dir'], 'val_guids.csv')
    val_df.to_csv(save_path, index=False)
    
    print(f"✅ 验证集GUID已保存: {save_path}")
    print(f"   样本数: {len(val_guids)}")
    print(f"   前5个GUID: {val_guids[:5]}")
    
    # 验证分布（可选）
    print("\n▶ 验证集类别分布:")
    labels = []
    for batch in val_loader:
        labels.extend(batch['label'].numpy().tolist())
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    for i in range(3):
        count = labels.count(i)
        print(f"   {label_map[i]:12s}: {count:4d} 样本 ({count/len(labels)*100:5.2f}%)")
    
    return val_guids

def main():
    """Spyder主入口 - 直接运行此函数"""
    # 1. 设置随机种子
    set_seed(CONFIG['seed'])
    
    # 2. 初始化训练器
    trainer = Trainer(CONFIG)
    
    # 保存验证集用
    # save_validation_guids(CONFIG)
    # sys.exit(0)
    
    # 3. 执行任务
    if CONFIG['mode'] in ['train', 'train_test']:
        trainer.train()
    
    if CONFIG['mode'] in ['test', 'train_test']:
        trainer.test()
    
    print("\n🎉 任务完成! 模型保存在输出目录:")
    print(f"   模型: {os.path.join(CONFIG['output_dir'], 'best_model.pth')}")
    print(f"   预测: {os.path.join(CONFIG['output_dir'], 'submission.csv')}")
    


if __name__ == "__main__":
    main()
    