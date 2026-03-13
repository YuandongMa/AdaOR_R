"""
Title: AdaOR: Oculomotor-Guided Perceptual Reconstruction for Extreme Sparse-View Imaging

Description:
-------------------------------------
Biologically interpretable feature extractor for sparse-view reconstruction.
It organizes the pipeline into a cortical-style hierarchy:
    - V1: Local geometric response
    - V2/V3: Cross-view structural integration
    - MEGM: Macro Eye-Movement Grouping Mechanism
    - MSRM: Micro Saccade Rhythm Mechanism with an internal hidden-state Router
    - V4: Global semantic abstraction
    - IT: Holistic integration

Inside MSRM, the Router explicitly decomposes hidden states into:
    - texture hidden state
    - structure hidden state
    - spatial hidden state

It predicts:
    - h_p in R^{N x 1} to filter spatially informative tokens
    - gamma in R^{3} to adaptively weight the three hidden branches
-------------------------------------
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing MambaBlock if available, else define fallback.
try:
    from mamba_ssm import MambaBlock  # type: ignore
except Exception:
    class MambaBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)


class V1_PixelCompletion(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class V2V3_StructureInference(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        out = x_flat + self.ffn(attn_out)
        return out.permute(0, 2, 1).reshape(b, c, h, w)


class V4_SemanticAbstraction(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mamba = MambaBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        out = self.mamba(x_flat)
        return out.permute(0, 2, 1).reshape(b, c, h, w)


class IT_HolisticIntegration(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(x.flatten(2), dim=2)
        out = self.fc(pooled)
        return out.unsqueeze(-1).unsqueeze(-1).expand_as(x)


class MEGM(nn.Module):
    """A lightweight grouping block for macroscopic saliency partitioning."""

    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.norms = nn.ModuleList([
            nn.GroupNorm(1, 1) for _ in range(num_groups)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = torch.chunk(x, self.num_groups, dim=1)
        processed = [norm(g) for norm, g in zip(self.norms, groups)]
        return torch.cat(processed, dim=1)


class HiddenStateRouter(nn.Module):
    """
    Router inside MSRM.

    Input:
        x: [B, C, H, W]

    Output:
        h_tex, h_str, h_spa: branch-specific hidden states, all [B, C, H, W]
        hp: spatial selector, [B, N, 1]
        gamma: branch weights, [B, 3, 1]
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(dim // reduction, 8)

        self.texture_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.structure_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.spatial_proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.spatial_selector = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.branch_weight = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        n = h * w

        h_tex = self.texture_proj(x)
        h_str = self.structure_proj(x)
        h_spa = self.spatial_proj(x)

        spatial_tokens = h_spa.flatten(2).permute(0, 2, 1)  # [B, N, C]
        hp = self.spatial_selector(spatial_tokens)  # [B, N, 1]

        pooled = x.mean(dim=(2, 3))  # [B, C]
        gamma = F.softmax(self.branch_weight(pooled), dim=-1).unsqueeze(-1)  # [B, 3, 1]

        hp_map = hp.permute(0, 2, 1).reshape(b, 1, h, w)
        h_spa = h_spa * hp_map

        assert hp.shape == (b, n, 1)
        assert gamma.shape == (b, 3, 1)

        return h_tex, h_str, h_spa, hp, gamma


class MSRM(nn.Module):
    """
    Micro Saccade Rhythm Mechanism.

    This module now contains an internal Router that disentangles the hidden
    representation into texture, structure, and spatial branches before
    branch-weighted fusion and attention-based rhythmic integration.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.saccade = nn.Conv2d(dim, dim, 3, padding=1)
        self.fixation = nn.Identity()
        self.microsaccade = nn.Conv2d(dim, dim, 1)
        self.router = HiddenStateRouter(dim)
        self.pre_fuse_norm = nn.BatchNorm2d(dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self, x: torch.Tensor, return_router_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.saccade(x) + self.microsaccade(x) + self.fixation(x)

        h_tex, h_str, h_spa, hp, gamma = self.router(x)
        fused = (
            gamma[:, 0].view(-1, 1, 1, 1) * h_tex
            + gamma[:, 1].view(-1, 1, 1, 1) * h_str
            + gamma[:, 2].view(-1, 1, 1, 1) * h_spa
        )
        fused = self.pre_fuse_norm(fused)

        b, c, h, w = fused.shape
        fused_tokens = fused.flatten(2).permute(0, 2, 1)
        attn_out, _ = self.attn(fused_tokens, fused_tokens, fused_tokens)
        attn_out = self.out_proj(attn_out)
        out = attn_out.permute(0, 2, 1).reshape(b, c, h, w)

        if return_router_stats:
            return out, {
                "hp": hp,
                "gamma": gamma,
                "texture_hidden": h_tex,
                "structure_hidden": h_str,
                "spatial_hidden": h_spa,
            }
        return out


class BioVisionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 64):
        super().__init__()
        self.v1 = V1_PixelCompletion(in_channels, dim)
        self.megm = MEGM(num_groups=4)
        self.v2v3 = V2V3_StructureInference(dim)
        self.msrm = MSRM(dim)
        self.v4 = V4_SemanticAbstraction(dim)
        self.it = IT_HolisticIntegration(dim)

    def forward(
        self, x: torch.Tensor, return_router_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.v1(x)
        x = self.megm(x)
        x = self.v2v3(x)

        if return_router_stats:
            x, stats = self.msrm(x, return_router_stats=True)
        else:
            x = self.msrm(x)
            stats = None

        x = self.v4(x)
        x = self.it(x)

        if return_router_stats:
            return x, stats  # type: ignore[return-value]
        return x
