import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, BertModel, BertTokenizer, ViTModel, ViTImageProcessor
from typing import Dict, Tuple

class ProjectionHead(nn.Module):
    """论文Section 3.1: 三类投影头 (Pimg, Ptxt, PCLIP)"""
    def __init__(self, input_dim: int, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CMSA(nn.Module):
    """论文Section 3.2.1: Cross-modal Self-Augmenting Attention (8层堆叠)"""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.self_attn_img = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.self_attn_txt = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.cross_attn_i2t = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.cross_attn_t2i = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 初始特征: [B, L_img, D], [B, L_txt, D]
        i, t = img_feat, txt_feat
        
        for layer_idx in range(self.num_layers):
            # Step 1: 模态内自注意力 (论文Fig.3 MSA)
            i_self, _ = self.self_attn_img[layer_idx](i, i, i)
            t_self, _ = self.self_attn_txt[layer_idx](t, t, t)
            i = self.norm(i + self.dropout(i_self))
            t = self.norm(t + self.dropout(t_self))
            
            # Step 2: 交叉注意力 (论文Fig.3 MCA)
            i_cross, _ = self.cross_attn_t2i[layer_idx](i, t, t)  # image query, text key/value
            t_cross, _ = self.cross_attn_i2t[layer_idx](t, i, i)  # text query, image key/value
            i = self.norm(i + self.dropout(i_cross))
            t = self.norm(t + self.dropout(t_cross))
        
        return i, t  # [B, L_img, D], [B, L_txt, D]

class CWA(nn.Module):
    """论文Section 3.2.2: CLIP-Weighted Attention Module"""
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        img_feat_cmsa: torch.Tensor,  # CMSA输出 [B, L_img, D]
        txt_feat_cmsa: torch.Tensor,  # CMSA输出 [B, L_txt, D]
        img_feat_clip: torch.Tensor,  # CLIP图像特征 [B, D]
        txt_feat_clip: torch.Tensor   # CLIP文本特征 [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: 计算CLIP相似度 (论文Eq.9)
        img_norm = F.normalize(img_feat_clip, p=2, dim=-1)
        txt_norm = F.normalize(txt_feat_clip, p=2, dim=-1)
        sim_clip = torch.sum(img_norm * txt_norm, dim=-1).unsqueeze(-1)  # [B, 1], 范围[-1,1]
        sim_clip = (sim_clip + 1) / 2  # 归一化到[0,1]
        
        # Step 2: 动态加权 (论文Eq.10)
        # 对序列特征取mean pooling得到全局表示再加权
        img_global = img_feat_cmsa.mean(dim=1)  # [B, D]
        txt_global = txt_feat_cmsa.mean(dim=1)  # [B, D]
        
        img_weighted = img_global * sim_clip
        txt_weighted = txt_global * sim_clip
        
        return img_weighted, txt_weighted  # [B, D], [B, D]

class CSFI(nn.Module):
    """论文Section 3.2.3: CLIP Self-Enhanced Feature Integration"""
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.proj_fusion = ProjectionHead(feat_dim * 2, feat_dim)  # PCLIP-mrg
        # 自监督解码器 (论文Eq.14)
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
    
    def forward(
        self, 
        img_clip: torch.Tensor,  # [B, D]
        txt_clip: torch.Tensor   # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: 特征拼接与投影 (论文Eq.12)
        fused = torch.cat([img_clip, txt_clip], dim=-1)  # [B, 2D]
        fused_feat = self.proj_fusion(fused)  # [B, D]
        
        # Step 2: 自监督重构 (论文Eq.13-14)
        recon_feat = self.decoder(fused_feat)  # [B, D]
        
        return fused_feat, recon_feat  # FCLIP-mrg, FCLIP-rec

class MAFL(nn.Module):
    """论文Section 3.2.4: Multimodal Attention Fusion Layer"""
    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.att_fc = nn.Sequential(
            nn.Linear(4, 64),  # 4种模态特征
            nn.GELU(),
            nn.Linear(64, 4)
        )
    
    def forward(
        self,
        txt_map: torch.Tensor,      # [B, D]
        img_map: torch.Tensor,      # [B, D]
        clip_fused: torch.Tensor,   # [B, D]
        clip_recon: torch.Tensor    # [B, D]
    ) -> torch.Tensor:
        # 构建特征矩阵 M = [Tmap, Vmap, FCLIP-mrg, FCLIP-rec] (论文Fig.6)
        M = torch.stack([txt_map, img_map, clip_fused, clip_recon], dim=1)  # [B, 4, D]
        
        # 平均池化 + 最大池化 (论文Eq.15)
        avg_pool = M.mean(dim=-1)  # [B, 4]
        max_pool = M.max(dim=-1)[0]  # [B, 4]
        pool_feat = avg_pool + max_pool  # [B, 4]
        
        # 生成注意力权重
        weights = torch.sigmoid(self.att_fc(pool_feat))  # [B, 4]
        weights = weights.unsqueeze(-1)  # [B, 4, 1]
        
        # 加权融合 (论文Eq.16)
        fused = (M * weights).sum(dim=1)  # [B, D]
        return fused

class CDAN(nn.Module):
    """完整CDAN模型 (论文Fig.2)"""
    def __init__(
        self,
        config,
        num_classes: int = 3,
        feat_dim: int = 512,
        cmsa_layers: int = 8,
        cmsa_heads: int = 8,
    ):
        super().__init__()
        # ===== 特征提取器 (冻结参数) =====
        clip_path = config.get('clip_path', 'openai/clip-vit-base-patch32')
        bert_path = config.get('bert_path', 'bert-base-uncased')
        vit_path = config.get('vit_path', 'google/vit-base-patch16-224')
        
        self.clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
        
        self.vit = ViTModel.from_pretrained(vit_path, local_files_only=True)
        self.vit_processor = ViTImageProcessor.from_pretrained(vit_path, local_files_only=True)
        
        # 冻结预训练参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # ===== 投影头 =====
        self.proj_img = ProjectionHead(768, feat_dim)   # ViT输出768维
        self.proj_txt = ProjectionHead(768, feat_dim)   # BERT输出768维
        self.proj_clip_img = ProjectionHead(512, feat_dim)  # CLIP图像特征512维
        self.proj_clip_txt = ProjectionHead(512, feat_dim)  # CLIP文本特征512维
        
        # ===== 核心融合模块 =====
        self.cmsa = CMSA(embed_dim=feat_dim, num_heads=cmsa_heads, num_layers=cmsa_layers)
        self.cwa = CWA()
        self.csfi = CSFI(feat_dim=feat_dim)
        self.mafl = MAFL(feat_dim=feat_dim)
        
        # ===== 分类器 =====
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # ===== 损失函数 =====
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.recon_loss_fn = nn.MSELoss()
    
    def extract_features(
        self, 
        images: torch.Tensor, 
        texts: list
    ) -> Dict[str, torch.Tensor]:
        """论文Section 3.1: 多模态特征提取"""
        # 单模态特征 (ViT + BERT)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_vit = (images - imagenet_mean) / imagenet_std
        vit_outputs = self.vit(pixel_values=images_vit)
        img_vit = vit_outputs.last_hidden_state[:, 0]  # [CLS] token
        img_vit_proj = self.proj_img(img_vit)
        
        bert_inputs = self.bert_tokenizer(
            texts, 
            padding=True,
            truncation=True,
            return_tensors="pt", 
            max_length=128
        ).to(images.device)
        bert_outputs = self.bert(**bert_inputs)
        txt_bert = bert_outputs.last_hidden_state[:, 0]  # [CLS] token
        txt_bert_proj = self.proj_txt(txt_bert)
        
        # CLIP跨模态特征
        clip_inputs = self.clip_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            max_length=77,         
            truncation=True,        
            do_rescale=False 
        ).to(images.device)
        clip_outputs = self.clip_model(**clip_inputs)
        img_clip = clip_outputs.image_embeds  # [B, 512]
        txt_clip = clip_outputs.text_embeds   # [B, 512]
        
        img_clip_proj = self.proj_clip_img(img_clip)
        txt_clip_proj = self.proj_clip_txt(txt_clip)
        
        return {
            "img_vit_proj": img_vit_proj.unsqueeze(1),  # [B, 1, D] 适配CMSA
            "txt_bert_proj": txt_bert_proj.unsqueeze(1),  # [B, 1, D]
            "img_clip": img_clip,
            "txt_clip": txt_clip,
            "img_clip_proj": img_clip_proj,
            "txt_clip_proj": txt_clip_proj
        }
    
    def forward(
        self,
        images: torch.Tensor,
        texts: list,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Step 1: 特征提取
        feats = self.extract_features(images, texts)
        
        # Step 2: CMSA模态交互 (论文Section 3.2.1)
        img_cmsa, txt_cmsa = self.cmsa(
            feats["img_vit_proj"], 
            feats["txt_bert_proj"]
        )  # [B, 1, D], [B, 1, D]
        
        # Step 3: CWA动态加权 (论文Section 3.2.2)
        img_cwa, txt_cwa = self.cwa(
            img_cmsa, txt_cmsa,
            feats["img_clip"], feats["txt_clip"]
        )  # [B, D], [B, D]
        
        # Step 4: CSFI特征增强 (论文Section 3.2.3)
        clip_fused, clip_recon = self.csfi(
            feats["img_clip_proj"], 
            feats["txt_clip_proj"]
        )  # [B, D], [B, D]
        
        # Step 5: MAFL自适应融合 (论文Section 3.2.4)
        fused_feat = self.mafl(
            txt_cwa, img_cwa, clip_fused, clip_recon
        )  # [B, D]
        
        # Step 6: 情感分类
        logits = self.classifier(fused_feat)  # [B, 3]
        
        output = {"logits": logits, "features": fused_feat}
        
        # 计算损失 (论文Section 3.3)
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits, labels)
            recon_loss = self.recon_loss_fn(clip_fused, clip_recon)
            total_loss = ce_loss + 0.1 * recon_loss  # λ=0.1 (论文Eq.17)
            output.update({
                "loss": total_loss,
                "ce_loss": ce_loss,
                "recon_loss": recon_loss
            })
        
        return output