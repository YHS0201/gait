import torch
import torch.nn.functional as F
import math
from .base import BaseLoss, gather_and_scale_wrapper

class CircleLoss(BaseLoss):
    def __init__(self,
                 margin=0.25,
                 gamma=256.0,
                 loss_term_weight=1.0,
                 l_min=-1.0,
                 l_max=1.0,
                 cycles=4,
                 cycle_iters=120000,
                 eps=1e-8,
                 grad_log_freq=100):  # 新增梯度记录频率参数
        super(CircleLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.gamma = gamma
        self.l_min = l_min
        self.l_max = l_max
        self.cycles = cycles
        self.cycle_iters = cycle_iters
        self.eps = eps
        self.grad_log_freq = grad_log_freq  # 控制梯度输出频率
        self.register_buffer('iter_count', torch.tensor(0, dtype=torch.long))
        
        # 新增梯度监控变量
        self.grad_info = {
            'weights_grad': None,
            'loss_grad': None,
            'base_losses_grad': None
        }

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        embeddings: [n, c, p] -> [p, n, c]
        labels    : [n]
        return    : (loss_avg, info)
        """
        # 开启梯度追踪用于后续分析
        with torch.set_grad_enabled(True):
            x = embeddings.permute(2, 0, 1).contiguous().float()  # [p, n, c]
            P = embeddings.size(2)
            
            # 逐部件计算Circle Loss
            losses = []
            for feat in x:
                loss_i = self._circle_single(feat, labels)
                losses.append(loss_i)
            base_losses = torch.stack(losses)  # [p]
            base_losses.retain_grad()  # 保留梯度用于后续分析
            
            # 动态指数调整（公式5）
            t = self.iter_count
            T = self.cycle_iters
            phase = 0.5 * math.pi * self.cycles * (t % T) / T
            l = (self.l_max - self.l_min) * torch.sin(phase) - self.l_min
            
            # Norm-Fusion权重计算（公式4）
            F_i_norm = base_losses / (torch.linalg.norm(base_losses, ord=2) + self.eps)
            a_i = torch.exp(F_i_norm)
            a_i_norm = a_i / (torch.linalg.norm(a_i, ord=2) + self.eps)
            weights = (a_i_norm.detach()) ** l

            
            # 加权损失融合
            weighted_loss = weights * base_losses
            loss = weighted_loss.sum()
            
            # 损失标准化（公式7）
            L_l0 = base_losses.sum()
            m = L_l0 / (loss + self.eps)
            loss_norm = (m / P) * loss
            loss_norm.retain_grad()  # 保留梯度用于后续分析

        # 更新迭代计数
        self.iter_count += 1
        
        # 梯度监控（按频率记录）
        if self.iter_count % self.grad_log_freq == 0:
            # 计算当前梯度
            self._capture_gradients(weights, base_losses, loss_norm)
            
            # 梯度分析（范数计算）
            grad_analysis = {
                'base_losses_grad_norm': torch.norm(self.grad_info['base_losses_grad']).item() if self.grad_info['base_losses_grad'] is not None else 0,
                'loss_grad_norm': torch.norm(self.grad_info['loss_grad']).item() if self.grad_info['loss_grad'] is not None else 0,
                'gradient_explosion': any(torch.isnan(g).any() or torch.isinf(g).any() 
                                         for g in self.grad_info.values() if g is not None)
            }
        else:
            grad_analysis = {}

        # 更新监控信息
        self.info.update({
            'loss': loss_norm.detach().clone(),
            'l_value': l.item(),
            'grad_analysis': grad_analysis
        })
        
        return loss_norm, self.info

    def _capture_gradients(self, weights, base_losses, loss_norm):
        
        if base_losses.grad is not None:
            self.grad_info['base_losses_grad'] = base_losses.grad.clone()
        if loss_norm.grad is not None:
            self.grad_info['loss_grad'] = loss_norm.grad.clone()

    def _circle_single(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        
        # 余弦相似度矩阵
        feat = F.normalize(feat, dim=1)
        sim = feat @ feat.T  # [n, n]

        # 构造正负掩码
        label_mat = label.unsqueeze(1) == label.unsqueeze(0)
        pos_mask = label_mat.triu(diagonal=1).bool()
        neg_mask = label_mat.logical_not().triu(diagonal=1).bool()

        if not (pos_mask.any() and neg_mask.any()):
            return torch.tensor(0.0, device=feat.device)

        sp = sim[pos_mask]  # 正样本相似度
        sn = sim[neg_mask]  # 负样本相似度

        # Circle Loss公式
        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))
        return loss