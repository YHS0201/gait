import torch
import torch.nn as nn
from collections import OrderedDict
from contextlib import nullcontext
from .videomae.utils.utils import load_state_dict
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model
from ..base_model import BaseModel
from ..modules import PackSequenceWrapper, HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks, BasicBlockP3D
from .videomae.models import modeling_pretrain  # noqa: F401 ensures model registration

import sys
import os


class VideoGait(BaseModel):
    
    def build_network(self, model_cfg):
        # 1. 构建ViT主干，指定img_size为(256,128)
        img_size = model_cfg.get('img_size', (256, 128))
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        elif isinstance(img_size, list):
            img_size = tuple(img_size)
        self.img_size_h, self.img_size_w = img_size
        self.patch_size = int(model_cfg.get('patch_size', 16))
        self.tubelet_size = int(model_cfg.get('tubelet_size', 2))
        all_frames = model_cfg.get('num_frames', 30)
        pretrained_name = model_cfg.get('pretrain_arch', 'pretrain_videomae_base_patch16_224')
        load_pretrained = model_cfg.get('load_pretrained_encoder', False)
        freeze_encoder = model_cfg.get('freeze_encoder', False)
        full_videomae = create_model(
            pretrained_name,
            pretrained=False,
            img_size=(self.img_size_h, self.img_size_w),
            all_frames=all_frames,
            tubelet_size=self.tubelet_size,
            patch_size=self.patch_size,
            encoder_depth=model_cfg.get('encoder_depth'),
        )
        self.vit = full_videomae.encoder
        self._encoder_num_patches = self.vit.patch_embed.num_patches
        # 编码器深度已在 create_model 中通过参数配置，无需再截断

        # 2. 根据需要加载预训练权重
        if load_pretrained:
            ckpt_path = model_cfg.get('videomae_ckpt')
            if not ckpt_path:
                raise ValueError('load_pretrained_encoder=True 但未提供 videomae_ckpt 路径')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            checkpoint_model = ckpt['model']
            cleaned_state_dict = OrderedDict()
            for k, v in checkpoint_model.items():
                key = k
                if key.startswith('module.'):
                    key = key[len('module.') :]
                cleaned_state_dict[key] = v

            load_msg = full_videomae.load_state_dict(cleaned_state_dict, strict=False)
            print(f"成功加载编码器权重")
            print(f"缺失的权重（应为空或仅有head）: {load_msg.missing_keys}")
            print(f"忽略的多余权重（应为decoder、mask_token等）: {load_msg.unexpected_keys}")
        else:
            print('VideoMAE 编码器将从随机初始化开始训练。')

        # 3. 根据配置决定是否冻结主干参数
        if freeze_encoder:
            self.vit.eval()
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            self.vit.train()
            for param in self.vit.parameters():
                param.requires_grad = True
        self._freeze_encoder = freeze_encoder

        # 4. 初始化其他组件
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        
        self.adapter = BasicBlockP3D(self.vit.embed_dim, self.vit.embed_dim, stride=1)
        
        # 5. 计算patch grid
        self.H_patches = self.img_size_h // self.patch_size
        self.W_patches = self.img_size_w // self.patch_size
        print(f"VideoMAE config: img_size=({self.img_size_h},{self.img_size_w}), patch_grid={self.H_patches}x{self.W_patches}")
        if self._freeze_encoder:
            print('编码器被冻结，不会参与梯度更新。')
        else:
            print('编码器处于训练状态，将参与梯度更新。')
        
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        # 极简版：按你的数据源格式直接统一到 [n, c, s, h, w]
        # 假设 rgbs 是 [n, s, c, h, w]，剪影 sils 是 [n, s, h, w]
        rgbs = rearrange(ipts[0].float(), 'n s c h w -> n c s h w')
        sils = ipts[1].float().unsqueeze(1)  # [n, 1, s, h, w]
        N, C, T, H, W = rgbs.shape
        # 将剪影转为0/1，并在通道维广播到RGB
        mask = (sils > 0).to(rgbs.dtype)
        mask = mask.repeat(1, C, 1, 1, 1) if mask.shape[1] == 1 else mask
        inputs_to_vit = rgbs * mask

        # 可视化：反归一化到像素空间并做前景相乘，输出到 visual_summary
        try:
            mean = torch.tensor([0.485*255, 0.456*255, 0.406*255], device=rgbs.device, dtype=rgbs.dtype).view(1, 3, 1, 1, 1)
            std = torch.tensor([0.229*255, 0.224*255, 0.225*255], device=rgbs.device, dtype=rgbs.dtype).view(1, 3, 1, 1, 1)
            rgbs_orig = rgbs * std + mean  # 回到像素值空间（0~255）
            fg_orig = rgbs_orig * mask
            fg_vis = torch.clamp(fg_orig / 255.0, 0.0, 1.0)
        except Exception:
            fg_vis = torch.clamp(inputs_to_vit, 0.0, 1.0)

        # 特征提取
        context = torch.no_grad() if self._freeze_encoder else nullcontext()
        with context:
            visible_mask = torch.zeros(
                inputs_to_vit.size(0),
                self._encoder_num_patches,
                device=inputs_to_vit.device,
                dtype=torch.bool,
            )
            outs = self.vit.forward_features(inputs_to_vit, visible_mask)
            outs0 = outs  # 保留原始 tokens，供 PCA 使用
            #print(outs[0])
            #sys.exit(0)
        # 重组为时空结构
        
        # 计算时间维的 patch 数，后续用于还原空间网格与池化
        T_patches = T // self.tubelet_size
        outs = rearrange(outs, 'b (t h w) c -> b c t h w',
                         t=T_patches, h=self.H_patches, w=self.W_patches)
        # 基于特征的可视化：按通道均值得到每帧的 patch 激活强度，上采样到原始分辨率，并映射为彩色
        try:
            outs_heat = outs.abs().mean(dim=1)  # (B, t, h, w)
            outs_heat_up = outs_heat.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
            bmin = outs_heat_up.amin(dim=(1,2,3), keepdim=True)
            bmax = outs_heat_up.amax(dim=(1,2,3), keepdim=True)
            outs_heat_norm = torch.clamp((outs_heat_up - bmin) / (bmax - bmin + 1e-6), 0.0, 1.0)
            Bv, Tv, Hv, Wv = outs_heat_norm.shape
            # 伪彩映射：R=强度，G=0，B=1-强度，形状 (B*t, 3, H, W)
            heat = outs_heat_norm.view(Bv * Tv, 1, Hv, Wv)
            zeros = torch.zeros_like(heat)
            outs_heatmap_vis = torch.cat([heat, zeros, 1.0 - heat], dim=1)

            # 叠加到前景：将输入前景按 tubelet 时间步采样与热力图对齐，然后按 alpha 混合
            # 对齐时间：取每个 tubelet 的首帧（步长为 tubelet_size），形状 (B, 3, T_patches, H, W)
            try:
                fg_tokens = fg_vis[:, :, ::self.tubelet_size, :, :]
                fg_tokens_bt = rearrange(fg_tokens, 'n c t h w -> (n t) c h w')
                # 若两者帧数不一致，取最小长度对齐
                min_bt = min(fg_tokens_bt.size(0), outs_heatmap_vis.size(0))
                alpha = 0.6
                overlay = alpha * outs_heatmap_vis[:min_bt] + (1 - alpha) * fg_tokens_bt[:min_bt]
                overlay = torch.clamp(overlay, 0.0, 1.0)
                outs_overlay_vis = overlay
            except Exception:
                outs_overlay_vis = None
        except Exception:
            outs_heatmap_vis = None
            outs_overlay_vis = None

        # PCA 可视化：直接使用原始 tokens (outs0)，通道维降到3并还原为空间网格
        try:
            # 组装所有 token 到二维矩阵 (N_tokens, C)，进行中心化
            X = outs0.detach().reshape(-1, outs0.size(-1))  # (B*T*H*W, C)
            X = X - X.mean(dim=0, keepdim=True)
            # 优先使用 torch.pca_lowrank，失败则回退 SVD
            try:
                U, S, V = torch.pca_lowrank(X, q=3)
                P = V[:, :3]
            except Exception:
                Vh = torch.linalg.svd(X, full_matrices=False).Vh
                P = Vh[:3, :].T
            Y = X @ P  # (N_tokens, 3)
            BT = outs0.size(0) * T_patches
            Y_img = Y.view(BT, self.H_patches, self.W_patches, 3).permute(0, 3, 1, 2).contiguous()  # (B*T,3,h,w)
            Y_up = Y_img.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
            ymin = Y_up.amin(dim=(1,2,3), keepdim=True)
            ymax = Y_up.amax(dim=(1,2,3), keepdim=True)
            outs_pca_vis = torch.clamp((Y_up - ymin) / (ymax - ymin + 1e-6), 0.0, 1.0)
        except Exception as e:
            print(f"PCA 可视化失败: {e}")
            outs_pca_vis = None
        
        
        
        
        outs = self.adapter(outs)
        
        
        outs = self.TP(outs, seqL, options={"dim": 2})[0]

        # 水平金字塔池化
        feat = self.HPP(outs)

        # FC和BNNeck
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
                'image/inputs_to_vit': rearrange(fg_vis, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {'embeddings': embed}
        }
        if 'outs_heatmap_vis' in locals() and outs_heatmap_vis is not None:
            retval['visual_summary']['image/outs_heatmap'] = outs_heatmap_vis
        if 'outs_overlay_vis' in locals() and outs_overlay_vis is not None:
            retval['visual_summary']['image/outs_overlay'] = outs_overlay_vis
        if 'outs_pca_vis' in locals() and outs_pca_vis is not None:
            retval['visual_summary']['image/outs_pca'] = outs_pca_vis
        return retval