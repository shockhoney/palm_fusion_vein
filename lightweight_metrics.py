import torch
import torch.nn as nn
from thop import profile
from models.stage1_mobileFacenet import MobileFaceNet
from models.stage2 import Stage2Fusion
from models import edgenext

class FullBiometricSystem(nn.Module):
    def __init__(self, input_size=224, embed_dim=256):
        super(FullBiometricSystem, self).__init__()
        # self.palm_net = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        # self.vein_net = MobileFaceNet(input_channel=3, input_size=input_size, embedding_size=embed_dim)
        # self.fusion_net = Stage2Fusion(in_dim_global=embed_dim, out_dim_final=512)
        self.palm_net = edgenext.EdgeNeXt(in_chans=3, num_classes=500,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5]).to('cpu')
        self.vein_net = edgenext.EdgeNeXt(in_chans=3, num_classes=500,
                 depths=[3, 3, 9, 3], dims=[24, 48, 88, 168],
                 global_block=[0, 0, 0, 3], global_block_type=['None', 'None', 'None', 'SDTA'],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., expan_ratio=4,
                 kernel_sizes=[7, 7, 7, 7], heads=[8, 8, 8, 8], use_pos_embd_xca=[False, False, False, False],
                 use_pos_embd_global=False, d2_scales=[2, 3, 4, 5]).to('cpu')
        self.fusion_net = Stage2Fusion(in_dim_global=168, out_dim_final=512).to('cpu')
    def forward(self, img_palm, img_vein):

        feat_palm = self.palm_net(img_palm, return_spatial=False)
        feat_vein = self.vein_net(img_vein, return_spatial=False)   
        final_feat = self.fusion_net(feat_palm, feat_vein)
        
        return final_feat


def calculate_efficiency():

    INPUT_SIZE = 224
    EMBED_DIM = 256  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FullBiometricSystem(input_size=INPUT_SIZE, embed_dim=EMBED_DIM).to(device)
    model.eval()

    # Batch Size 设为 1，这是计算推理 FLOPs 的标准做法
    dummy_palm = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    dummy_vein = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    # inputs 必须是一个 tuple，对应 forward 函数的参数
    flops, params = profile(model, inputs=(dummy_palm, dummy_vein), verbose=False)

    GFLOPs = flops / 1e9
    Params_M = params / 1e6
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