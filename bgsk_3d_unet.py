"""
Boundary-Guided Selective Kernel 3D U-Net for Glioma Segmentation
Implementation of Boundary-Guided SK mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryGuidedSKConv3D(nn.Module):
    """
    Boundary-Guided Selective Kernel Convolution for 3D
    
    Innovation: Modulates SK attention based on boundary detection
    - At boundaries: Prefer small kernels (fine details)
    - At center: Prefer large kernels (context)
    """
    def __init__(self, in_ch, out_ch, M=2, r=16, boundary_weight=0.5):
        super().__init__()
        self.M = M
        self.boundary_weight = boundary_weight
        d = max(out_ch // r, 32)
        
        # Two branches with different receptive fields
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1),  # 3×3×3
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),  # 5×5×5 effective
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Standard SK attention mechanism
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(out_ch, d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv3d(d, out_ch, kernel_size=1) for _ in range(M)])
        
        # NEW: Boundary detection module
        self.boundary_detector = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output: [0, 1], high at boundaries
        )
    
    def forward(self, x):
        # Step 1: Apply two branches
        U = [conv(x) for conv in self.convs]  # [U1, U2]
        U_sum = sum(U)
        
        # Step 2: Standard SK attention (content-based)
        s = self.gap(U_sum)  # [B, C, 1, 1, 1]
        z = self.relu(self.fc1(s))  # [B, d, 1, 1, 1]
        
        # Compute base attention weights
        attention_logits = [fc(z) for fc in self.fc2]  # List of [B, C, 1, 1, 1]
        attention_base = torch.softmax(torch.stack(attention_logits, dim=1), dim=1)  # [B, M, C, 1, 1, 1]
        
        # Step 3: NEW - Boundary-guided modulation
        boundary_map = self.boundary_detector(x)  # [B, 1, D, H, W]
        
        # Expand boundary_map to match attention dimensions
        boundary_map = boundary_map.unsqueeze(2)  # [B, 1, 1, D, H, W]
        
        # Modulate attention based on boundary
        # Branch 0 (small kernel): Boost at boundaries
        attention_0 = attention_base[:, 0] * (1 + self.boundary_weight * boundary_map)
        # Branch 1 (large kernel): Reduce at boundaries
        attention_1 = attention_base[:, 1] * (1 - self.boundary_weight * boundary_map)
        
        # Renormalize
        attention_stack = torch.stack([attention_0, attention_1], dim=1)  # [B, M, C, D, H, W]
        attention_sum = attention_stack.sum(dim=1, keepdim=True)
        attention_normalized = attention_stack / (attention_sum + 1e-8)
        
        # Step 4: Aggregate branches with modulated attention
        V = sum([attention_normalized[:, i] * U[i] for i in range(self.M)])
        
        return V


class BGSKBlock3D(nn.Module):
    """Boundary-Guided SK Block with two BG-SK convolutions"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bgsk1 = BoundaryGuidedSKConv3D(in_ch, out_ch)
        self.bgsk2 = BoundaryGuidedSKConv3D(out_ch, out_ch)
    
    def forward(self, x):
        x = self.bgsk1(x)
        x = self.bgsk2(x)
        return x


class BGSK3DUNet(nn.Module):
    """
    Boundary-Guided Selective Kernel 3D U-Net
    
    Main Innovation: BG-SK modules that adapt kernel selection based on boundary information
    """
    def __init__(self, in_channels=4, out_channels=5):
        super().__init__()
        
        # Encoder with BG-SK blocks
        self.enc1 = BGSKBlock3D(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = BGSKBlock3D(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = BGSKBlock3D(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = BGSKBlock3D(128, 256)
        
        # Decoder with BG-SK blocks
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = BGSKBlock3D(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = BGSKBlock3D(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = BGSKBlock3D(64 + 32, 32)
        
        # Output
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Output
        out = self.out(d1)
        
        return out


# Test function
if __name__ == "__main__":
    # Test BG-SK module
    print("Testing Boundary-Guided SK Conv3D...")
    bgsk = BoundaryGuidedSKConv3D(in_ch=4, out_ch=32)
    x = torch.randn(2, 4, 32, 32, 32)  # [B, C, D, H, W]
    out = bgsk(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in bgsk.parameters()):,}")
    
    print("\n" + "="*60)
    
    # Test full model
    print("Testing BG-SK 3D U-Net...")
    model = BGSK3DUNet(in_channels=4, out_channels=5)
    x = torch.randn(1, 4, 64, 64, 64)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✅ All tests passed!")
