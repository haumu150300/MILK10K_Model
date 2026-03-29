import torch
import torch.nn as nn

class ASLWithLogitsLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=2.0, clip=0.05, eps=1e-6):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        x = x.clamp(min=self.eps, max=1 - self.eps)
        
        xs_pos = x
        xs_neg = 1 - x

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        log_pos = torch.log(xs_pos.clamp(min=self.eps))
        log_neg = torch.log(xs_neg.clamp(min=self.eps))

        with torch.no_grad():
            pos_weight = torch.pow(1 - xs_pos, self.gamma_pos)
            neg_weight = torch.pow(xs_pos, self.gamma_neg)

        pos_loss = targets * log_pos * pos_weight
        neg_loss = (1 - targets) * log_neg * neg_weight

        loss = - (pos_loss + neg_loss).mean()
        return loss