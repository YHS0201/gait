import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D
from utils import get_attr_from, get_valid_args


blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
    '3d': BasicBlock3D
}


class DeepGaitV2Recon(BaseModel):

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map.keys()
        block = blocks_map[mode]

        mask_cfg = model_cfg.get('recon', {}) if isinstance(model_cfg, dict) else {}
        self.input_mask_enabled = mask_cfg.get('enabled', True)
        self.input_mask_ratio = mask_cfg.get('mask_ratio', 0.9)
        self.lambda_recon = float(mask_cfg.get('lambda', 1.0))
        self.lambda_edge = float(mask_cfg.get('lambda_edge', 1.0))
        self.edge_sobel_thr_ratio = float(mask_cfg.get('edge_sobel_thr_ratio', 0.2))
        self.edge_sobel_thr_abs = float(mask_cfg.get('edge_sobel_thr_abs', 0.0))

        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

        if mode == '3d':
            strides = [
                [1, 1],
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 1]
            ]
        else:
            strides = [
                [1, 1],
                [2, 2],
                [2, 2],
                [1, 1]
            ]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        ))
        self.layer1 = SetBlockWrapper(self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode=mode))

        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.decoder = nn.Sequential(
            nn.Conv3d(channels[2], 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1)
        )
        self.decoder_edge = nn.Sequential(
            nn.Conv3d(channels[2], 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1)
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def make_layer(self, block, planes, stride, blocks_num, mode='2d'):

        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=stride, padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride=stride), nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=[1, 1, 1], stride=[1, *stride], padding=[0, 0, 0], bias=False), nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError('xxx')
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1, 1] if mode in ['2d', 'p3d'] else [1, 1, 1]
        for i in range(1, blocks_num):
            layers.append(
                    block(self.inplanes, planes, stride=s)
            )
        return nn.Sequential(*layers)

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver', 'backbone_lr', 'decoder_lr'])

        backbone_lr = float(optimizer_cfg.get('backbone_lr', optimizer_cfg['lr']))
        decoder_lr = float(optimizer_cfg.get('decoder_lr', optimizer_cfg['lr']))

        backbone_modules = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.FCs,
            self.BNNecks,
        ]
        params_list = [
            {
                'params': [p for module in backbone_modules for p in module.parameters() if p.requires_grad],
                'lr': backbone_lr,
            },
            {
                'params': [p for p in self.decoder.parameters() if p.requires_grad],
                'lr': decoder_lr,
            },
            {
                'params': [p for p in self.decoder_edge.parameters() if p.requires_grad],
                'lr': decoder_lr,
            },
        ]
        optimizer = optimizer(params_list, **valid_arg)
        return optimizer

    def _build_edge_target(self, sils):
        B, _, T, H, W = sils.shape
        x2d = sils.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 1, H, W)

        kx = torch.tensor([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], device=x2d.device, dtype=x2d.dtype).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 2.0, 1.0]], device=x2d.device, dtype=x2d.dtype).view(1, 1, 3, 3)

        gx = F.conv2d(x2d, kx, padding=1)
        gy = F.conv2d(x2d, ky, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)

        max_per_frame = grad_mag.flatten(2).amax(dim=2, keepdim=True).view(B * T, 1, 1, 1)
        thr = self.edge_sobel_thr_ratio * max_per_frame + self.edge_sobel_thr_abs
        sobel_edge = (grad_mag > thr).to(x2d.dtype)

        sil_bin = (x2d > 0.0).to(x2d.dtype)
        dilate = F.max_pool2d(sil_bin, kernel_size=3, stride=1, padding=1)
        erode = 1.0 - F.max_pool2d(1.0 - sil_bin, kernel_size=3, stride=1, padding=1)
        morph_edge = (dilate - erode).clamp(0.0, 1.0)

        edge = torch.maximum(sobel_edge, morph_edge)
        edge = edge.view(B, T, 1, H, W).permute(0, 2, 1, 3, 4).contiguous()
        return edge

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0]
            sils = sils.transpose(1, 2).contiguous()
        assert sils.size(-1) in [44, 88]

        orig_sils = sils

        if self.training and self.input_mask_enabled:
            ratio = float(self.input_mask_ratio)
            ratio = max(0.0, min(ratio, 1.0))
            B, _, S, H, W = sils.shape
            ph, pw = 4, 4
            if (H % ph != 0) or (W % pw != 0):
                raise ValueError(f"Masking requires input divisible by 4x4. Got H={H}, W={W}.")

            grid_h = H // ph
            grid_w = W // pw
            num_patches = grid_h * grid_w

            fg0 = (sils[:, :, 0] > 0).to(torch.float32)
            fg_patch = fg0.view(B, 1, grid_h, ph, grid_w, pw).mean(dim=(3, 5))
            fg_patch_flat = fg_patch.view(B, num_patches)
            fg_thr = 0.05
            bg_eps = 1e-2
            power = 2.0

            patch_mask_flat = torch.zeros(B, num_patches, device=sils.device, dtype=sils.dtype)
            for b in range(B):
                fg_mask = fg_patch_flat[b] > fg_thr
                num_fg = int(fg_mask.sum().item())

                if num_fg <= 0:
                    num_mask_b = int(ratio * num_patches)
                    num_mask_b = max(0, min(num_mask_b, num_patches))
                    if num_mask_b > 0:
                        idx = torch.randperm(num_patches, device=sils.device)[:num_mask_b]
                        patch_mask_flat[b, idx] = 1.0
                    continue

                num_mask_b = int(ratio * num_fg)
                num_mask_b = max(0, min(num_mask_b, num_fg))
                if num_mask_b <= 0:
                    continue

                fg_w = (fg_patch_flat[b] + bg_eps).pow(power)
                fg_w = fg_w * fg_mask.to(fg_w.dtype)
                if float(fg_w.sum()) <= 0.0:
                    fg_w = fg_mask.to(fg_w.dtype)
                if float(fg_w.sum()) > 0.0:
                    fg_idx = torch.multinomial(fg_w, num_mask_b, replacement=False)
                    patch_mask_flat[b, fg_idx] = 1.0

            pm = patch_mask_flat.view(B, 1, grid_h, grid_w)
            spatial_mask = pm.repeat_interleave(ph, dim=2).repeat_interleave(pw, dim=3)
            spatial_mask_t = spatial_mask.unsqueeze(2).expand(B, 1, S, H, W)
            masked_sils = sils * (1.0 - spatial_mask_t)
        else:
            masked_sils = sils

        del ipts
        out0 = self.layer0(masked_sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        outs = self.TP(out4, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)

        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)

        if self.inference_use_emb2:
            embed = embed_2
        else:
            embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed
            }
        }

        if self.training and self.input_mask_enabled:
            recon = self.decoder(out3)
            edge_logits = self.decoder_edge(out3)

            B, _, S, H0, W0 = orig_sils.shape
            _, _, Sd, Hd, Wd = recon.shape
            recon_up = F.interpolate(
                recon.view(B * Sd, 1, Hd, Wd), size=(H0, W0), mode='bilinear', align_corners=False
            ).view(B, 1, Sd, H0, W0)
            edge_logits_up = F.interpolate(
                edge_logits.view(B * Sd, 1, Hd, Wd), size=(H0, W0), mode='bilinear', align_corners=False
            ).view(B, 1, Sd, H0, W0)

            assert S == Sd, f"Temporal mismatch: input S={S}, recon Sd={Sd}"
            T = S
            recon_up = recon_up[:, :, :T]
            edge_logits_up = edge_logits_up[:, :, :T]
            target = orig_sils[:, :, :T]
            edge_target = self._build_edge_target(target).to(target.dtype)

            m = spatial_mask.unsqueeze(2).expand(B, 1, T, H0, W0)
            diff2 = (recon_up - target) ** 2
            denom = m.sum().clamp_min(1.0)
            recon_mse = (diff2 * m).sum() / denom
            edge_bce_map = F.binary_cross_entropy_with_logits(edge_logits_up, edge_target, reduction='none')
            edge_loss = (edge_bce_map * m).sum() / denom

            retval['training_feat']['recon_mse'] = self.lambda_recon * recon_mse
            retval['training_feat']['edge_bce'] = self.lambda_edge * edge_loss
            retval['visual_summary']['image/recon'] = rearrange(recon_up.detach(), 'n c s h w -> (n s) c h w')
            retval['visual_summary']['image/recon_edge'] = rearrange(torch.sigmoid(edge_logits_up.detach()), 'n c s h w -> (n s) c h w')
            retval['visual_summary']['image/target_edge'] = rearrange(edge_target.detach(), 'n c s h w -> (n s) c h w')
            retval['visual_summary']['image/masked'] = rearrange(masked_sils[:, :, :T], 'n c s h w -> (n s) c h w')
            retval['visual_summary']['image/target'] = rearrange(target, 'n c s h w -> (n s) c h w')

        return retval