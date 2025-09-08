import torch
import torch.nn.functional as F
import math
from .base import BaseLoss, gather_and_scale_wrapper
from .tripletnew import NEWTripletLoss

class Waveloss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, margin=0.2, eps=0.1, loss_term_weight=1.0, log_accuracy=False, l_min=-1.0,
                 l_max=1.0,
                 cycles=4,
                 cycle_iters=120000,):
        super(Waveloss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        self.l_min = l_min
        self.l_max = l_max
        self.cycles = cycles
        self.cycle_iters = cycle_iters
        self.eps = eps
        self.register_buffer('iter_count', torch.tensor(0, dtype=torch.long))
        self.base_loss = NEWTripletLoss(margin=margin, loss_term_weight=1.0)
        
    @gather_and_scale_wrapper
    def forward(self, embeddings, logits, labels):
            
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        P, N, C = embeddings.shape
        
        base_losses = []
        for i in range(P):
            loss, _ = self.base_loss(embeddings[i].unsqueeze(0), labels)
            base_losses.append(loss)
        base_losses= torch.stack(base_losses)
            
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            row_loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps, reduction='none')
        else:
            row_loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p), reduction='none')
        base_losses = base_losses + row_loss.mean(dim=0)
            
        t = self.iter_count
        T = self.cycle_iters
        phase = 0.5 * math.pi * self.cycles * (t % T) / T
        l = (self.l_max - self.l_min) * torch.sin(phase) - self.l_min
        
        # 自适应权重计算
        F_i_norm = base_losses / (torch.linalg.norm(base_losses, ord=2) + 1e-8)
        a_i = torch.exp(F_i_norm)
        a_i_norm = a_i / (torch.linalg.norm(a_i, ord=2) + 1e-8)
        weights = (a_i_norm.detach()) ** l
        
        # 加权损失融合
        weighted_loss = weights * base_losses
        loss = weighted_loss.sum()
        
        # 损失标准化
        L_l0 = base_losses.sum()
        m = L_l0 / (loss + 1e-8)
        loss_norm = (m / p) * loss
        
        # 梯度截断
        

        # 更新迭代计数
        self.iter_count += 1

        self.info.update({'loss': loss_norm.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss_norm, self.info