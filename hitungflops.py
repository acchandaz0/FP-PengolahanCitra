# # measure_flops.py — run this separately after training
# import torch
# from fvcore.nn import FlopCountAnalysis
# from mmsk.mmsk_3d_unet_old import MMSK3DUNet
# from monai.networks.nets import UNet
# from brats_utils import OUT_CHANNELS

# # change model instantiation to whichever variant you want to measure
# model = MMSK3DUNet(in_channels=4, out_channels=OUT_CHANNELS, store_attention=False)
# model.eval()

# dummy = torch.zeros(1, 4, 128, 128, 128)
# flops = FlopCountAnalysis(model, dummy)
# flops.unsupported_ops_warnings(False)
# print(f"GFLOPs: {flops.total() / 1e9:.2f}")
# print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

"""
hitung_efisiensi.py — Hitung Parameter, GFLOPs, dan waktu inferensi
untuk tiap varian arsitektur. Jalankan di mesin yang punya file model.

Butuh: pip install thop
Jalankan:  python hitung_efisiensi.py
"""
import torch
import time

# ── ganti import sesuai file model kamu ────────────────────────────────────
# Baseline pakai MONAI UNet; Ablasi1 pakai BGSK; Ablasi2 & Usulan pakai MMSK
from monai.networks.nets import UNet
from bgsk_3d_unet_patched import BGSK3DUNet
from mmsk_3d_unet import MMSK3DUNet   # <- pastikan ini versi yang kamu pakai

try:
    from thop import profile
except ImportError:
    raise SystemExit("Install dulu: pip install thop")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = (1, 4, 128, 128, 128)   # sesuai preprocessing kamu (128^3, 4 modalitas)


def build_unet():
    return UNet(spatial_dims=3, in_channels=4, out_channels=5,
                channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2)


def measure(name, model):
    model = model.to(DEVICE).eval()
    x = torch.randn(*INPUT_SHAPE).to(DEVICE)

    # Parameter (trainable)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GFLOPs via thop
    with torch.no_grad():
        flops, _ = profile(model, inputs=(x,), verbose=False)
    gflops = flops / 1e9

    # Waktu inferensi rata-rata (20 run, buang 3 warmup)
    times = []
    with torch.no_grad():
        for i in range(23):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append((time.time() - t0) * 1000)  # ms
    avg_ms = sum(times) / len(times)

    print(f"{name:<14} | Param: {n_params:>12,} | GFLOPs: {gflops:8.2f} | "
          f"Inferensi: {avg_ms:7.2f} ms")
    return n_params, gflops, avg_ms


if __name__ == "__main__":
    print(f"Device: {DEVICE} | Input: {INPUT_SHAPE}")
    print("=" * 70)
    # Baseline1 & Baseline2 arsitektur identik (UNet) -> ukur sekali
    measure("Baseline (UNet)", build_unet())
    measure("Ablasi1 (BGSK)", BGSK3DUNet(in_channels=4, out_channels=5))
    measure("Ablasi2/Usulan (MMSK)",
            MMSK3DUNet(in_channels=4, out_channels=5, store_attention=False))
    print("=" * 70)
    print("Catatan: Baseline1 & Baseline2 sama (UNet). Ablasi2 & Usulan sama (MMSK).")
    print("GFLOPs & Param hanya bergantung arsitektur, jadi valid tanpa training.")
    print("Waktu inferensi bergantung mesin -- jalankan di mesin yang konsisten")
    print("untuk semua varian agar sebanding.")
