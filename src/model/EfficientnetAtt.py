import timm
from torch import nn
import torch
from torch.nn import functional as F


class AttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.normq = nn.LayerNorm(dim)
        self.normkv = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, dim)
        )

    def forward(self, fea_q, fea_kv=None):
        q = self.q_proj(self.normq(fea_q))
        if fea_kv is None:
            fea_kv = fea_q
        k = self.k_proj(self.normkv(fea_kv))
        v = self.v_proj(self.normkv(fea_kv))
        
        att_out, _ = self.att(q, k, v)
        x = fea_q + self.dropout(att_out)
        
        fnn_out = self.ffn(self.ffn_norm(x))
        x = x + self.dropout(fnn_out)
        return x
    
class EfficientnetAtt(nn.Module):
    def __init__(self, image_size=256, num_classes=11, extra_features=11):
        super().__init__()
        self.backbone1 = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', in_chans=3, pretrained=True, num_classes=0)
        # freeze backbone parameters
        for param in self.backbone1.parameters():
            param.requires_grad = False
        dim = self.backbone1.num_features
        
        self.fc = nn.Linear(dim*2, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)
        
        self.att = AttentionFusion(dim)
        self.pool = nn.Sequential(
           nn.Linear(dim, 1),
           nn.Dropout(0.5)
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, x2):
        fea1 = self.backbone1.forward_features(x)
        fea2 = self.backbone1.forward_features(x2)

        fea1 = fea1.flatten(2).transpose(1,2)
        fea2 = fea2.flatten(2).transpose(1,2)

        fused = torch.cat([fea1, fea2], dim=-1)
        fused = F.normalize(fused, dim=-1)
        
        fused = self.fc(fused)
        fused = self.norm(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        
        fused =  fused + self.att(fused)
        w = torch.softmax(self.pool(fused), dim=1)
        out = self.head((fused * w).sum(dim=1))
        return out


# import torch
# model = EfficientnetAtt()
# x = torch.randn(1, 3, 384, 384)

# out = model(x, x)
# print(out.shape)