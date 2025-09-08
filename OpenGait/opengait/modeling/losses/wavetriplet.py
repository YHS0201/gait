import torch
import torch.nn.functional as F
import math
from .base import BaseLoss, gather_and_scale_wrapper
from .tripletnew import NEWTripletLoss

class WaveLoss(BaseLoss):
    def __init__(self, 
                 loss_term_weight=1.0,
                 l_min=-1.0,
                 l_max=1.0,
                 cycles=4,
                 cycle_iters=120000,
                 eps=1e-8,
                 margin=0.2):
        super(WaveLoss, self).__init__(loss_term_weight)
        self.l_min = l_min
        self.l_max = l_max
        self.cycles = cycles
        self.cycle_iters = cycle_iters
        self.eps = eps
        self.register_buffer('iter_count', torch.tensor(0, dtype=torch.long))
        self.base_loss = NEWTripletLoss(margin=margin, loss_term_weight=1.0)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        P, N, C = embeddings.shape
        
        base_losses = []
        for i in range(P):
            loss, _ = self.base_loss(embeddings[i].unsqueeze(0), labels)
            base_losses.append(loss)
        base_losses= torch.stack(base_losses)
        
        # 动态计算指数 l
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
        
        # 梯度截断
        

        # 更新迭代计数
        self.iter_count += 1
        
        # 监控信息
        self.info.update({
            'loss': loss_norm.detach().clone()})
        
        return loss_norm, self.info