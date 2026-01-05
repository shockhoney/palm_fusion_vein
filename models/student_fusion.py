import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2FusionStudent_BottleneckGate(nn.Module):
    def __init__(self, in_dim_global=256, out_dim_final=512, bottleneck=128, gate_hidden=32, final_l2norm=True):
        super().__init__()
        self.final_l2norm = final_l2norm
        self.d = bottleneck

        self.adapter_p = nn.Linear(in_dim_global, bottleneck, bias=False)
        self.adapter_v = nn.Linear(in_dim_global, bottleneck, bias=False)
        self.norm_p = nn.LayerNorm(bottleneck)
        self.norm_v = nn.LayerNorm(bottleneck)

        self.gate = nn.Sequential(
            nn.Linear(4 * bottleneck, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2 * bottleneck),
        )
        self.proj = nn.Linear(bottleneck, out_dim_final)

    def forward(self, F_palm, F_vein):
        p = self.norm_p(self.adapter_p(F_palm))
        v = self.norm_v(self.adapter_v(F_vein))

        cross = torch.cat([p, v, p * v, (p - v).abs()], dim=1)  # [B,4d]
        w = self.gate(cross).view(p.size(0), 2, self.d)
        w = F.softmax(w, dim=1)

        fused = w[:, 0] * p + w[:, 1] * v                       # [B,d]
        fused = self.proj(fused)                                # [B,512]
        if self.final_l2norm:
            fused = F.normalize(fused, dim=1)
        return fused
