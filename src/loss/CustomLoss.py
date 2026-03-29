import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        
        freq = torch.tensor([
            303, 2522, 44, 544, 52,
            50, 9, 450, 746, 473, 47
        ], dtype=torch.float32).to(device)

        alpha = 1.0 / (freq + 1e-8)
        alpha = alpha / alpha.mean()
        self.alpha = alpha
        self.gamma_pos = 0
        self.gamma_neg = 4

    def forward(self, logits, targets, g=None):
        prob = torch.sigmoid(logits).clamp(1e-8, 1 - 1e-8)

        pos = targets
        neg = 1 - targets

        pos_loss = pos * torch.log(prob + 1e-8)
        neg_loss = neg * torch.log(1 - prob + 1e-8)

        pos_loss *= (1 - prob) ** self.gamma_pos
        neg_loss *= prob ** self.gamma_neg

        loss = - (self.alpha * pos_loss + neg_loss)
        loss = loss.mean()

        # 🔥 gate regularization
        if g is not None:
            loss_gate = (g.mean() - 0.5) ** 2
            loss = loss + 0.1 * loss_gate

        return loss