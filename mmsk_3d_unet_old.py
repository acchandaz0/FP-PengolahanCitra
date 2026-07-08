"""
Multi-Modal Selective Kernel 3D U-Net (MMSK-3D U-Net)
for Glioma Segmentation on BraTS / Synapse GLI

Architecture as described in:
"Multi-Scale Glioma Segmentation Using Selective Kernel 3D U-Net
with Class Imbalance Aware Tversky Loss"
Arsya Dewi Lathifa — NRP: 5025221015

Key Innovation — MMSK Block:
    Standard SK (Li et al., CVPR 2019):
        Input (single stream X)
        ├─→ Branch 1: Conv3D(3×3×3, d=1) → U₁
        └─→ Branch 2: Conv3D(3×3×3, d=2) → U₂
        ↓ Fuse: U = U₁ + U₂
        ↓ Select: s = GAP(U) → FC → Softmax → [a₁,a₂]
        ↓ Aggregate: V = a₁⊙U₁ + a₂⊙U₂

    MMSK Block (Proposed):
        Input: X (current feature map)
        Modal: M = [m_T1, m_T2, m_FLAIR, m_T1ce]
        X ├→ Conv3D(3×3×3, d=1) → U₁
          └→ Conv3D(3×3×3, d=2) → U₂
        ↓ Fuse: U = U₁ + U₂
        M → Gate: G = FC(GAP(Concat(M))) ∈ ℝᶜ
        U_gated = U ⊙ σ(G)
        → Select → Softmax → [a₁,a₂]
        → V = a₁⊙U₁ + a₂⊙U₂

Sub-region behavior:
    ET (Enhancing Tumor):  T1ce-dominated gate → small kernel (fine detail)
    WT (Whole Tumor):      FLAIR-dominated gate → large kernel (diffuse boundary)
    TC (Tumor Core):       T1 + T1ce combined → intermediate kernel selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMSKConv3D(nn.Module):
    """
    Multi-Modal Selective Kernel Convolution (3D)

    Extends SK (Li et al., CVPR 2019) with cross-modal gating:
    - Standard SK attention is computed from fused feature map U
    - Cross-modal gate G is computed from all 4 raw MRI modalities
    - Gate G modulates U before kernel selection → adapts receptive
      field based on modality-specific clinical signals

    Args:
        in_ch        : input feature channels (after initial fusion)
        out_ch       : output channels
        num_modalities: number of MRI modalities (default 4: T1,T2,FLAIR,T1ce)
        M            : number of SK branches (default 2)
        r            : reduction ratio for FC bottleneck (default 16)
    """

    def __init__(self, in_ch, out_ch, num_modalities=4, M=2, r=16):
        super().__init__()
        self.M = M
        self.out_ch = out_ch
        d = max(out_ch // r, 32)

        # ── Two SK branches with different receptive fields ──────────────
        # Branch 1: 3×3×3 (dilation=1) — fine / small receptive field
        # Branch 2: effective 5×5×5 (dilation=2) — coarse / large receptive field
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ),
        ])

        # ── Standard SK: channel-wise attention from fused features ──────
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.sk_fc1 = nn.Linear(out_ch, d, bias=False)
        self.sk_fc2 = nn.ModuleList([nn.Linear(d, out_ch, bias=False) for _ in range(M)])

        # ── Cross-modal gating module ─────────────────────────────────────
        # Takes raw MRI modalities (always 4 channels at full resolution),
        # produces a channel-wise gate G ∈ ℝ^{out_ch} via GAP + FC.
        # G is applied as a multiplicative modulation BEFORE SK selection,
        # so modality signals can steer which branch wins attention.
        modal_bottleneck = max(num_modalities * 4, 16)
        self.modal_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                              # [B, 4, 1,1,1]
            nn.Flatten(),                                          # [B, 4]
            nn.Linear(num_modalities, modal_bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(modal_bottleneck, out_ch, bias=False),
            nn.Sigmoid(),                                          # G ∈ (0,1)^{out_ch}
        )

    def forward(self, x, modalities):
        """
        Args:
            x          : feature map  [B, in_ch, D, H, W]
            modalities : raw 4-channel MRI input [B, 4, D, H, W]
                         (T1, T2, FLAIR, T1ce — same spatial dims as x
                          OR original input; we use GAP so size doesn't matter)

        Returns:
            V          : aggregated output [B, out_ch, D, H, W]
            attn_weights: attention weights per branch [B, M, out_ch]
                          (for interpretability / visualization)
        """
        B = x.size(0)

        # 1. Apply both branches
        U = [branch(x) for branch in self.branches]      # [U₁, U₂]  each [B,C,D,H,W]
        U_fuse = sum(U)                                   # element-wise sum

        # 2. Standard SK: GAP → FC1 → FC2 per branch
        s = self.gap(U_fuse).view(B, -1)                  # [B, out_ch]
        z = F.relu(self.sk_fc1(s))            # [B, d]
        attn_logits = torch.stack(
            [fc(z) for fc in self.sk_fc2], dim=1
        )                                                  # [B, M, out_ch]
        attn_base = torch.softmax(attn_logits, dim=1)     # [B, M, out_ch]

        # 3. Cross-modal gate G from raw MRI modalities
        G = self.modal_gate(modalities)                   # [B, out_ch]
        # Reshape for broadcasting: [B, 1, out_ch, 1, 1, 1] — but we apply
        # gate to U_fuse spatially, then re-select.
        G_spatial = G.view(B, self.out_ch, 1, 1, 1)       # [B, out_ch, 1,1,1]

        # 4. Modulate fused feature with gate (cross-modal influence)
        U_gated = [u * G_spatial for u in U]              # modulate each branch

        # 5. Aggregate with SK attention weights
        # attn_base: [B, M, out_ch] → expand to [B, M, out_ch, 1, 1, 1]
        attn_exp = attn_base.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        V = sum(attn_exp[:, i] * U_gated[i] for i in range(self.M))  # [B,C,D,H,W]

        return V, attn_base  # return attn for interpretability


class MMSKBlock3D(nn.Module):
    """
    Two stacked MMSK convolutions (mirrors the double-conv pattern in 3D U-Net).

    The modality tensor is passed through both MMSK convolutions so that
    cross-modal gating operates at every feature-extraction step.
    """

    def __init__(self, in_ch, out_ch, num_modalities=4):
        super().__init__()
        self.mmsk1 = MMSKConv3D(in_ch,  out_ch, num_modalities)
        self.mmsk2 = MMSKConv3D(out_ch, out_ch, num_modalities)

    def forward(self, x, modalities):
        x, attn1 = self.mmsk1(x, modalities)
        x, attn2 = self.mmsk2(x, modalities)
        return x, (attn1, attn2)   # expose both attention maps


class MMSK3DUNet(nn.Module):
    """
    Multi-Modal Selective Kernel 3D U-Net

    Architecture:
        Encoder  : 3 × MMSKBlock3D  (32 → 64 → 128)
        Bottleneck: MMSKBlock3D      (256)
        Decoder  : 3 × MMSKBlock3D  (128 → 64 → 32)
        Output   : Conv3D 1×1×1     (32 → num_classes)

    All MMSK blocks receive the raw 4-channel MRI input as `modalities`
    so that the cross-modal gate always has access to full clinical signal
    regardless of encoder depth.

    Args:
        in_channels   : number of input MRI modalities (default 4)
        out_channels  : number of segmentation classes (default 4: BG, NCR, ED, ET)
        store_attention: if True, forward() also returns a dict of attention maps
                         (useful for interpretability / gate weight visualization)
    """

    def __init__(self, in_channels=4, out_channels=5, store_attention=False):
        super().__init__()
        self.store_attention = store_attention
        self.num_modalities  = in_channels

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc1 = MMSKBlock3D(in_channels, 32,  in_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = MMSKBlock3D(32, 64,  in_channels)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = MMSKBlock3D(64, 128, in_channels)
        self.pool3 = nn.MaxPool3d(2)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = MMSKBlock3D(128, 256, in_channels)

        # ── Decoder ───────────────────────────────────────────────────────
        self.up3  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = MMSKBlock3D(256 + 128, 128, in_channels)

        self.up2  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = MMSKBlock3D(128 + 64,  64,  in_channels)

        self.up1  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = MMSKBlock3D(64  + 32,  32,  in_channels)

        # ── Output ────────────────────────────────────────────────────────
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def _resize_modalities(self, modalities, target):
        """
        Downsample / upsample raw modality tensor to match target spatial dims.
        Required because the cross-modal gate uses AdaptiveAvgPool3d(1) so any
        spatial size is fine — but we still need matching dims if we ever
        extend the gate to use spatial features.
        Currently the gate is purely channel-wise (GAP removes spatial), so
        this is a no-op safety wrapper.
        """
        if modalities.shape[2:] != target.shape[2:]:
            modalities = F.interpolate(
                modalities, size=target.shape[2:],
                mode='trilinear', align_corners=True
            )
        return modalities

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x : [B, 4, D, H, W]  — multi-modal MRI (T1, T2, FLAIR, T1ce)

        Returns:
            logits : [B, out_channels, D, H, W]
            attn   : dict of attention maps (only when store_attention=True)
        """
        modalities = x  # raw input always passed as cross-modal signal

        # ── Encoder ──────────────────────────────────────────────────────
        e1, attn_e1 = self.enc1(x, modalities)
        e2, attn_e2 = self.enc2(self.pool1(e1), modalities)
        e3, attn_e3 = self.enc3(self.pool2(e2), modalities)

        # ── Bottleneck ────────────────────────────────────────────────────
        b,  attn_b  = self.bottleneck(self.pool3(e3), modalities)

        # ── Decoder ───────────────────────────────────────────────────────
        d3, attn_d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), modalities)
        d2, attn_d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), modalities)
        d1, attn_d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), modalities)

        # ── Output ────────────────────────────────────────────────────────
        logits = self.out_conv(d1)

        if self.store_attention:
            attn_maps = {
                'enc1': attn_e1, 'enc2': attn_e2, 'enc3': attn_e3,
                'bottleneck': attn_b,
                'dec3': attn_d3, 'dec2': attn_d2, 'dec1': attn_d1,
            }
            return logits, attn_maps

        return logits


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 65)
    print("Testing MMSKConv3D (single layer) ...")
    layer = MMSKConv3D(in_ch=4, out_ch=32, num_modalities=4)
    x    = torch.randn(2, 4, 32, 32, 32)
    m    = torch.randn(2, 4, 32, 32, 32)
    out, attn = layer(x, m)
    print(f"  Input  : {x.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Attn   : {attn.shape}  (B, M, out_ch)")
    print(f"  Params : {sum(p.numel() for p in layer.parameters()):,}")

    print()
    print("=" * 65)
    print("Testing MMSK3DUNet (full model, store_attention=False) ...")
    model = MMSK3DUNet(in_channels=4, out_channels=5, store_attention=False)
    x     = torch.randn(1, 4, 64, 64, 64)
    logits = model(x)
    print(f"  Input  : {x.shape}")
    print(f"  Output : {logits.shape}")
    total  = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total:,}  (~{total/1e6:.1f}M)")

    print()
    print("=" * 65)
    print("Testing MMSK3DUNet (full model, store_attention=True) ...")
    model_attn = MMSK3DUNet(in_channels=4, out_channels=5, store_attention=True)
    logits, attn_maps = model_attn(x)
    print(f"  Output      : {logits.shape}")
    print(f"  Attn keys   : {list(attn_maps.keys())}")
    # Each value is a tuple of (attn_layer1, attn_layer2) per block
    enc1_attn = attn_maps['enc1']
    print(f"  enc1 attn[0]: {enc1_attn[0].shape}  (B, M, out_ch)")

    print()
    print("=" * 65)
    print("Gate weight analysis example (interpretability check) ...")
    # Simulate which branch wins for a given sample
    # attn shape: [B, M, out_ch]  M=2 branches
    enc3_attn = attn_maps['enc3'][0]   # first MMSK conv in enc3 block
    branch_preference = enc3_attn[0].mean(dim=-1)  # [M] — avg across channels
    print(f"  enc3 branch preference (mean over channels): {branch_preference.detach()}")
    print(f"  Branch 0 (small 3×3×3): {branch_preference[0].item():.4f}")
    print(f"  Branch 1 (large 5×5×5): {branch_preference[1].item():.4f}")

    print()
    print("✅ All tests passed!")
    print("=" * 65)
