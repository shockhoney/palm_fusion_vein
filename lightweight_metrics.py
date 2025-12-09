import torch
import torch.nn as nn
from thop import profile
from models.stage1_mobileFacenet import MobileFaceNet
from models.stage2 import Stage2Fusion

class FullBiometricSystem(nn.Module):
    def __init__(self, input_size=224, embed_dim=256):
        super(FullBiometricSystem, self).__init__()
        # 实例化两个骨干网络
        self.palm_net = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        self.vein_net = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        
        # 实例化融合模块 (根据你 train.py 里的配置)
        self.fusion_net = Stage2Fusion(in_dim_global=embed_dim, out_dim_final=512)

    def forward(self, img_palm, img_vein):
        # 模拟 train_phase2 中的前向传播过程
        
        # 1. 提取特征 (不返回空间特征，只返回 embedding)
        feat_palm = self.palm_net(img_palm, return_spatial=False)
        feat_vein = self.vein_net(img_vein, return_spatial=False)
        
        # 2. 特征融合
        final_feat = self.fusion_net(feat_palm, feat_vein)
        
        return final_feat

# ==========================================
# 2. 主计算逻辑
# ==========================================
def calculate_efficiency():
    # 配置参数
    INPUT_SIZE = 224
    EMBED_DIM = 256  # MobileFaceNet 默认输出 256
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化完整模型
    model = FullBiometricSystem(input_size=INPUT_SIZE, embed_dim=EMBED_DIM).to(device)
    model.eval()

    # 创建虚拟输入 (Dummy Input)
    # Batch Size 设为 1，这是计算推理 FLOPs 的标准做法
    dummy_palm = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    dummy_vein = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    print("正在计算 Parameters 和 FLOPs ...")

    # 使用 thop 进行统计
    # inputs 必须是一个 tuple，对应 forward 函数的参数
    flops, params = profile(model, inputs=(dummy_palm, dummy_vein), verbose=False)

    # ==========================================
    # 3. 结果转换与输出
    # ==========================================
    
    # FLOPs 转为 G (10^9)
    GFLOPs = flops / 1e9
    
    # Params 转为 M (10^6)
    Params_M = params / 1e6
    
    # 模型大小估算 (Model Size)
    # PyTorch 默认参数是 float32，一个参数占 4 bytes
    # 1 MB = 1024 * 1024 bytes
    Model_Size_MB = (params * 4) / (1024 * 1024)

    print("\n" + "="*40)
    print(" Model lightweighting Metrics")
    print("="*40)
    print(f"1. Parameters:  {Params_M:.4f} M ")
    print(f"2. FLOPs:       {GFLOPs:.4f} G ")
    print(f"3. Model Size: {Model_Size_MB:.4f} MB ")
    print("="*40)

    palm_params = sum(p.numel() for p in model.palm_net.parameters()) / 1e6
    fusion_params = sum(p.numel() for p in model.fusion_net.parameters()) / 1e6
    print(f" - single Backbone:   {palm_params:.4f} M")
    print(f" - Fusion:     {fusion_params:.4f} M")
    print("="*40)

if __name__ == "__main__":
    calculate_efficiency()