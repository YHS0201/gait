import torch
import torch.nn.functional as F
import math
from .base import BaseLoss, gather_and_scale_wrapper


class CircleLoss(BaseLoss):
    def __init__(self,
                 margin = 0.25,
                 gamma = 256.0,
                 loss_term_weight = 1.0,
                 l_min = -1.0,
                 l_max = 1.0,
                 cycles = 4,
                 cycle_iters = 120000,
                 eps = 1e-8):
        super(CircleLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.gamma = gamma
        self.l_min = l_min
        self.l_max = l_max
        self.cycles = cycles
        self.cycle_iters = cycle_iters
        self.eps = eps
        self.register_buffer('iter_count', torch.tensor(0, dtype=torch.long))

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        embeddings: [n, c, p] -> [p, n, c]
        labels    : [n]
        return    : (loss_avg, info)
        """
        x = embeddings.permute(2, 0, 1).contiguous().float()  # [p, n, c]
        P = embeddings.size(2)
        # 2. 逐部件计算 Circle Loss
        losses = []
        for feat in x:                      # feat: [n, c]
            loss_i = self._circle_single(feat, labels)
            losses.append(loss_i)
        base_losses = torch.stack(losses)        # [p]

        t = self.iter_count
        T = self.cycle_iters
        phase = 0.5 * math.pi * self.cycles * (t % T) / T
        l = (self.l_max - self.l_min) * torch.sin(phase) - self.l_min
        
        # 自适应权重计算
        F_i_norm = base_losses / (torch.linalg.norm(base_losses, ord=2) + self.eps)
        a_i = torch.exp(F_i_norm)
        a_i_norm = a_i / (torch.linalg.norm(a_i, ord=2) + self.eps)
        weights = (a_i_norm.detach()) ** l
        
        # 加权损失融合
        weighted_loss = weights * base_losses
        loss = weighted_loss.sum()
        
        # 损失标准化
        L_l0 = base_losses.sum()
        m = L_l0 / (loss + self.eps)
        loss_norm = (m / P) * loss
        
        #loss_norm.register_hook(
            #lambda g: print(f"[iter {self.iter_count.item()}] loss_grad={g.item():.4f}")
        #)
        #print('loss_norm:', loss_norm.item())

        # 更新迭代计数
        self.iter_count += 1
        
        # 监控信息
        self.info.update({
            'loss': loss_norm.detach().clone()})
        
        return loss_norm, self.info

    # ------------- 内部工具 -------------
    def _circle_single(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        feat : [n, c]
        label: [n]
        返回单个部件的 Circle Loss
        """
        # 余弦相似度矩阵
        feat = F.normalize(feat, dim=1)
        sim = feat @ feat.T                                # [n, n]

        # 构造正负掩码（上三角，不包含对角线）
        label_mat = label.unsqueeze(1) == label.unsqueeze(0)
        pos_mask = label_mat.triu(diagonal=1).bool()
        neg_mask = label_mat.logical_not().triu(diagonal=1).bool()

        if not (pos_mask.any() and neg_mask.any()):
            return torch.tensor(0.0, device=feat.device)

        sp = sim[pos_mask]    # 正样本相似度
        sn = sim[neg_mask]    # 负样本相似度

        # Circle Loss 公式
        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin,  min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n =  an * (sn - delta_n) * self.gamma

        loss = F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))
        return loss

