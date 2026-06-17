import argparse
import time

import torch
import torch.nn as nn
from thop import profile

from models.resnet18_encoder import ResNet18Encoder
from models.stage1_mobileFacenet import MobileFaceNet
from models.stage2 import Stage2Fusion


class FusionSystem(nn.Module):
    def __init__(self, model_type="student", input_size=224, embed_dim=256):
        super().__init__()
        encoder = ResNet18Encoder if model_type == "teacher" else MobileFaceNet
        self.palm_net = encoder(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        self.vein_net = encoder(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        self.fusion_net = Stage2Fusion(in_dim_global=embed_dim, out_dim_final=512, final_l2norm=True)

    def forward(self, palm, vein):
        palm_feat = self.palm_net(palm, return_spatial=False)
        vein_feat = self.vein_net(vein, return_spatial=False)
        return self.fusion_net(palm_feat, vein_feat)


@torch.no_grad()
def benchmark_latency(model, palm, vein, warmup, iters):
    for _ in range(warmup):
        model(palm, vein)
    if palm.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(palm, vein)
        if palm.is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    t = torch.tensor(times)
    return float(t.mean()), float(t.std(unbiased=False)), float(t.median()), float(torch.quantile(t, 0.95))


def main():
    parser = argparse.ArgumentParser("Measure complexity and latency")
    parser.add_argument("--model", choices=["teacher", "student"], default="student")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = FusionSystem(args.model, input_size=args.input_size).to(device).eval()
    palm = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    vein = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    flops, params = profile(model, inputs=(palm, vein), verbose=False)
    mean_ms, std_ms, p50_ms, p95_ms = benchmark_latency(model, palm, vein, args.warmup, args.iters)

    print(f"model,{args.model}")
    print(f"device,{device}")
    print(f"batch_size,{args.batch_size}")
    print(f"params_M,{params / 1e6:.4f}")
    print(f"flops_G,{flops / 1e9:.4f}")
    print(f"model_size_MB,{params * 4 / (1024 * 1024):.4f}")
    print(f"latency_mean_ms,{mean_ms:.4f}")
    print(f"latency_std_ms,{std_ms:.4f}")
    print(f"latency_p50_ms,{p50_ms:.4f}")
    print(f"latency_p95_ms,{p95_ms:.4f}")


if __name__ == "__main__":
    main()
