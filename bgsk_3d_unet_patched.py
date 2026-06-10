"""
Boundary-Guided Selective Kernel 3D U-Net — Opsi 2 (deadline-optimized)

PERUBAHAN DARI ORIGINAL:
    Masalah asal: 14 boundary_detector calls per forward pass
    → 7 BGSKBlock3D × 2 BoundaryGuidedSKConv3D = 14 Conv3d ekstra
    → Ini yang membuat 1 epoch = 31 menit (vs Ablation 2 = 2.4 menit)

    Opsi 2: boundary_detector HANYA di enc1 (layer pertama, resolusi penuh).
    enc2, enc3, bottleneck, dec3, dec2, dec1 → pakai PlainSKConv3D
    (standard SK tanpa boundary detector).

    Estimasi speedup:
        Original : 14 BD calls → ~31 menit/epoch
        Opsi 2   : 2 BD calls  → ~12–13 menit/epoch
        → 300 epoch ≈ 60–65 jam  (masih terlalu lama)
        → dengan early stopping patience=10: bisa selesai 80–120 epoch
          = 16–26 jam dari sekarang

JUSTIFIKASI ARSITEKTUR (untuk thesis/paper):
    Boundary cues paling informatif ada di enc1 karena:
    - Resolusi spasial penuh (128³) → gradien tepi belum terdilusi pooling
    - Input langsung dari MRI → boundary_detector belajar dari raw intensitas
    - Layer dalam (enc2+) sudah punya receptive field besar; SK attention
      di sana lebih dari cukup untuk konteks semantik

BUGFIX yang juga disertakan:
    - [FIX] Hapus boundary_map.unsqueeze(2) yang menyebabkan output 6D
      → conv3d crash dengan "Expected 5D input, got 6D"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# PlainSKConv3D — standard Selective Kernel tanpa boundary detector
# Dipakai oleh enc2, enc3, bottleneck, dec3, dec2, dec1
# ─────────────────────────────────────────────────────────────────────────────
class PlainSKConv3D(nn.Module):
    """
    Standard 3D Selective Kernel convolution (dua branch, tanpa boundary detector).
    Menggantikan BoundaryGuidedSKConv3D di semua layer selain enc1.
    """
    def __init__(self, in_ch, out_ch, M=2, r=16):
        super().__init__()
        self.M = M
        d = max(out_ch // r, 32)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
        ])

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(out_ch, d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.ModuleList([nn.Conv3d(d, out_ch, kernel_size=1) for _ in range(M)])

    def forward(self, x):
        U = [conv(x) for conv in self.convs]          # each [B, C, D, H, W]
        U_sum = sum(U)                                  # [B, C, D, H, W]

        s = self.gap(U_sum)                             # [B, C, 1, 1, 1]
        z = self.relu(self.fc1(s))                      # [B, d, 1, 1, 1]

        attn_logits = [fc(z) for fc in self.fc2]        # list of [B, C, 1, 1, 1]
        attn = torch.softmax(
            torch.stack(attn_logits, dim=1), dim=1
        )  # [B, M, C, 1, 1, 1]

        V = sum(attn[:, i] * U[i] for i in range(self.M))  # [B, C, D, H, W]
        return V


# ─────────────────────────────────────────────────────────────────────────────
# BoundaryGuidedSKConv3D — SK + boundary detector
# HANYA dipakai oleh enc1
# ─────────────────────────────────────────────────────────────────────────────
class BoundaryGuidedSKConv3D(nn.Module):
    """
    SK + boundary detector. Digunakan HANYA di enc1 (resolusi penuh 128³).

    BUGFIX: boundary_map.unsqueeze(2) DIHAPUS.
        boundary_map  : [B, 1, D, H, W]   → 5D
        attn_base[:,0]: [B, C, 1, 1, 1]   → 5D
        Produk        : [B, C, D, H, W]   → 5D ✓
        unsqueeze(2) sebelumnya membuat [B,1,1,D,H,W] (6D) → crash conv3d.
    """
    def __init__(self, in_ch, out_ch, M=2, r=16, boundary_weight=0.5):
        super().__init__()
        self.M = M
        self.boundary_weight = boundary_weight
        d = max(out_ch // r, 32)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
        ])

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(out_ch, d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.ModuleList([nn.Conv3d(d, out_ch, kernel_size=1) for _ in range(M)])

        self.boundary_detector = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Branch convolutions
        U = [conv(x) for conv in self.convs]           # each [B, C, D, H, W]
        U_sum = sum(U)

        # Standard SK attention
        s = self.gap(U_sum)                             # [B, C, 1, 1, 1]
        z = self.relu(self.fc1(s))                      # [B, d, 1, 1, 1]
        attn_logits = [fc(z) for fc in self.fc2]
        attn_base = torch.softmax(
            torch.stack(attn_logits, dim=1), dim=1
        )  # [B, M, C, 1, 1, 1]

        # Boundary-guided modulation
        boundary_map = self.boundary_detector(x)        # [B, 1, D, H, W]
        # PENTING: JANGAN unsqueeze(2) di sini.
        # attn_base[:,0] adalah [B, C, 1, 1, 1]
        # boundary_map   adalah [B, 1, D, H, W]
        # Broadcasting → [B, C, D, H, W] ✓  (5D, aman untuk conv3d berikutnya)

        attn_0 = attn_base[:, 0] * (1 + self.boundary_weight * boundary_map)
        attn_1 = attn_base[:, 1] * (1 - self.boundary_weight * boundary_map)

        attn_stack = torch.stack([attn_0, attn_1], dim=1)       # [B, M, C, D, H, W]
        attn_norm  = attn_stack / (attn_stack.sum(dim=1, keepdim=True) + 1e-8)

        V = sum(attn_norm[:, i] * U[i] for i in range(self.M))  # [B, C, D, H, W]
        return V


# ─────────────────────────────────────────────────────────────────────────────
# Blocks
# ─────────────────────────────────────────────────────────────────────────────
class BGSKBlock3D(nn.Module):
    """enc1: dua BoundaryGuidedSKConv3D"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bgsk1 = BoundaryGuidedSKConv3D(in_ch, out_ch)
        self.bgsk2 = BoundaryGuidedSKConv3D(out_ch, out_ch)

    def forward(self, x):
        return self.bgsk2(self.bgsk1(x))


class PlainSKBlock3D(nn.Module):
    """enc2–dec1: dua PlainSKConv3D (tanpa boundary detector)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.sk1 = PlainSKConv3D(in_ch, out_ch)
        self.sk2 = PlainSKConv3D(out_ch, out_ch)

    def forward(self, x):
        return self.sk2(self.sk1(x))


# ─────────────────────────────────────────────────────────────────────────────
# BGSK3DUNet — Opsi 2
# ─────────────────────────────────────────────────────────────────────────────
class BGSK3DUNet(nn.Module):
    """
    BGSK 3D U-Net — Opsi 2 (boundary detector hanya di enc1).

    Drop-in replacement untuk original BGSK3DUNet.
    Interface identik: BGSK3DUNet(in_channels=4, out_channels=5)
    """
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()

        # Encoder
        self.enc1  = BGSKBlock3D(in_channels, 32)   # ← BD aktif (resolusi penuh)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2  = PlainSKBlock3D(32, 64)          # ← plain SK
        self.pool2 = nn.MaxPool3d(2)

        self.enc3  = PlainSKBlock3D(64, 128)         # ← plain SK
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = PlainSKBlock3D(128, 256)   # ← plain SK

        # Decoder
        self.up3  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = PlainSKBlock3D(256 + 128, 128)  # ← plain SK

        self.up2  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = PlainSKBlock3D(128 + 64, 64)    # ← plain SK

        self.up1  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = PlainSKBlock3D(64 + 32, 32)     # ← plain SK

        # Output
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Smoke test: BoundaryGuidedSKConv3D (bugfix check) ===")
    bgsk = BoundaryGuidedSKConv3D(in_ch=4, out_ch=32)
    x = torch.randn(2, 4, 32, 32, 32)
    out = bgsk(x)
    assert out.ndim == 5, f"Harus 5D, dapat {out.ndim}D"
    assert out.shape == (2, 32, 32, 32, 32), f"Shape salah: {out.shape}"
    print(f"  Input : {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}  ✓")

    print("\n=== Smoke test: PlainSKConv3D ===")
    psk = PlainSKConv3D(in_ch=64, out_ch=64)
    x2 = torch.randn(2, 64, 16, 16, 16)
    out2 = psk(x2)
    assert out2.shape == (2, 64, 16, 16, 16)
    print(f"  Input : {tuple(x2.shape)}")
    print(f"  Output: {tuple(out2.shape)}  ✓")

    print("\n=== Smoke test: BGSK3DUNet full forward (opsi 2) ===")
    model = BGSK3DUNet(in_channels=4, out_channels=5)
    x3 = torch.randn(1, 4, 64, 64, 64)
    out3 = model(x3)
    assert out3.shape == (1, 5, 64, 64, 64), f"Shape salah: {out3.shape}"
    print(f"  Input : {tuple(x3.shape)}")
    print(f"  Output: {tuple(out3.shape)}  ✓")
    total = sum(p.numel() for p in model.parameters())
    bd    = sum(p.numel() for p in model.enc1.parameters())
    print(f"  Total params  : {total:,}")
    print(f"  enc1 params   : {bd:,}  (bagian yang punya boundary detector)")
    print(f"  Rasio BD/total: {bd/total*100:.1f}%")
    print("\n✅ Semua test passed!")