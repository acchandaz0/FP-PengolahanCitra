# measure_flops.py — run this separately after training
import torch
from fvcore.nn import FlopCountAnalysis
from mmsk_3d_unet import MMSK3DUNet
from monai.networks.nets import UNet
from brats_utils import OUT_CHANNELS

# change model instantiation to whichever variant you want to measure
model = MMSK3DUNet(in_channels=4, out_channels=OUT_CHANNELS, store_attention=False)
model.eval()

dummy = torch.zeros(1, 4, 128, 128, 128)
flops = FlopCountAnalysis(model, dummy)
flops.unsupported_ops_warnings(False)
print(f"GFLOPs: {flops.total() / 1e9:.2f}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
