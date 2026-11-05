"""
RESPIRA - Methodology Model Architecture
=========================================
Implements the exact deep learning architecture from the intended methodology:
  1. Dual Backbones:
     - EfficientNet-B0 -> Spatial maps [B, 1280, 16, 16] -> 256 tokens of dim 1280
     - Vision Transformer (ViT-B/16) -> 1024 visual tokens of dim 768 (at 512x512)
  2. Feature Projection:
     - CNN tokens: Linear(1280, 512) + LayerNorm + GELU
     - ViT tokens: Linear(768, 512) + LayerNorm + GELU
  3. Bidirectional Cross-Attention:
     - CNN queries ViT context (Q=CNN, K=V=ViT) -> [B, 256, 512]
     - ViT queries CNN context (Q=ViT, K=V=CNN) -> [B, 1024, 512]
     - Concatenation -> [B, 1280, 512]
  4. Disease-Conditioned Attention & Adaptive Fusion:
     - 6 learnable disease queries Q [6, 512] attend to 1280 fused tokens (K,V)
     - Produces 6 dedicated disease feature vectors [B, 6, 512]
  5. Disease Relationship Modeling (Label Correlation / GCN):
     - Self-attention across the 6 disease channels to produce adjacency matrix A [B, 6, 6]
     - Relational Graph Convolution message propagation: H_rel = bmm(A, H) + H
     - 6 dedicated disease classification heads -> Logits [B, 6]
  6. Uncertainty Quantification:
     - Monte Carlo (MC) Dropout inference: T stochastic forward passes
     - Predictive mean probability, epistemic variance, and normalized binary entropy
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

TARGET_CLASSES = [
    "Atelectasis",
    "Bacterial Pneumonia",
    "Normal",
    "Pulmonary Edema",
    "Tuberculosis",
    "Viral Pneumonia",
]
NUM_CLASSES = 6


# ============================================================
# 1. CROSS-ATTENTION BLOCK
# ============================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Multi-head cross-attention
        attended, weights = self.attn(
            query=query, key=key_value, value=key_value, need_weights=True
        )
        x = self.norm1(query + attended)
        out = self.norm2(x + self.ffn(x))
        return out, weights


# ============================================================
# 2. BIDIRECTIONAL CROSS-ATTENTION
# ============================================================
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # CNN queries ViT context
        self.cnn_to_vit = CrossAttentionBlock(embed_dim, num_heads, dropout)
        # ViT queries CNN context
        self.vit_to_cnn = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(
        self, cnn_tokens: torch.Tensor, vit_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cnn_tokens: [B, 256, 512], vit_tokens: [B, 1024, 512]
        cnn_enh, c2v_att = self.cnn_to_vit(cnn_tokens, vit_tokens)
        vit_enh, v2c_att = self.vit_to_cnn(vit_tokens, cnn_tokens)
        # Concatenate along sequence dimension -> [B, 1280, 512]
        fused = torch.cat([cnn_enh, vit_enh], dim=1)
        return fused, c2v_att, v2c_att


# ============================================================
# 3. DISEASE-CONDITIONED ATTENTION
# ============================================================
class DiseaseConditionedAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        # 6 learnable disease query embeddings
        self.disease_queries = nn.Parameter(
            torch.randn(num_classes, embed_dim) * 0.02
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, fused_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = fused_tokens.shape[0]
        # Expand queries across batch: [B, 6, 512]
        queries = self.disease_queries.unsqueeze(0).expand(B, -1, -1)
        attended, weights = self.attn(
            query=queries,
            key=fused_tokens,
            value=fused_tokens,
            need_weights=True,
        )
        x = self.norm1(queries + attended)
        disease_features = self.norm2(x + self.ffn(x))  # [B, 6, 512]
        return disease_features, weights


# ============================================================
# 4. DISEASE RELATIONSHIP MODELING (GCN)
# ============================================================
class DiseaseRelationshipGCN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention to discover correlation matrix between the 6 diseases
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Node projection
        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # GCN update transformation
        self.gcn_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self, disease_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # disease_features: [B, 6, 512]
        _, adj_matrix = self.self_attn(
            disease_features, disease_features, disease_features, need_weights=True
        )  # [B, 6, 6]

        # Row-normalize adjacency matrix
        adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-6)

        # Graph message passing
        H0 = self.node_proj(disease_features)  # [B, 6, 256]
        H_rel = torch.bmm(adj_norm, H0)  # Message propagation [B, 6, 256]
        H_enhanced = H0 + self.gcn_transform(H_rel)  # Residual graph update [B, 6, 256]

        return H_enhanced, adj_norm


# ============================================================
# 5. DEDICATED DISEASE HEADS
# ============================================================
class DedicatedDiseaseHeads(nn.Module):
    def __init__(self, in_features: int = 256, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        self.dropout_prob = dropout
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(128, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [B, 6, 256]
        logits = []
        for i, head in enumerate(self.heads):
            # Extract feature vector corresponding to disease i: [B, 256]
            feat_i = node_features[:, i, :]
            logit_i = head(feat_i)  # [B, 1]
            logits.append(logit_i)
        return torch.cat(logits, dim=1)  # [B, 6]


# ============================================================
# 6. MASTER RESPIRA METHODOLOGY MODEL
# ============================================================
class RespiraMethodologyModel(nn.Module):
    """
    Complete end-to-end multi-modal architecture with MC Dropout support.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ----------------------------------------------------
        # Stream 1: EfficientNet-B0 Backbone
        # ----------------------------------------------------
        weights_eff = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = efficientnet_b0(weights=weights_eff)
        self.cnn_backbone = eff.features  # Outputs [B, 1280, 16, 16] for 512x512
        self.cnn_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Stream 2: Vision Transformer (ViT-B/16) Backbone
        # ----------------------------------------------------
        weights_vit = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit_backbone = vit_b_16(weights=weights_vit)
        self.vit_backbone.heads = nn.Identity()

        # Positional embedding interpolation for 512x512 (1024 patch tokens)
        self._interpolate_vit_pos_embed()

        self.vit_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # ----------------------------------------------------
        # Fusion Modules
        # ----------------------------------------------------
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=512, num_heads=8, dropout=0.1
        )
        self.disease_attention = DiseaseConditionedAttention(
            embed_dim=512, num_classes=num_classes, num_heads=8, dropout=0.1
        )
        self.gcn_relationship = DiseaseRelationshipGCN(
            embed_dim=512, hidden_dim=256, num_classes=num_classes, num_heads=8, dropout=0.1
        )
        self.heads = DedicatedDiseaseHeads(
            in_features=256, num_classes=num_classes, dropout=dropout
        )

    def _interpolate_vit_pos_embed(self):
        """Bicubic interpolation of ViT positional embeddings from 14x14 (224) to 32x32 (512)."""
        pos_embed = self.vit_backbone.encoder.pos_embedding  # [1, 197, 768]
        cls_pos = pos_embed[:, :1, :]  # [1, 1, 768]
        patch_pos = pos_embed[:, 1:, :]  # [1, 196, 768]

        dim = patch_pos.shape[-1]
        grid_old = int(math.sqrt(patch_pos.shape[1]))  # 14
        grid_new = 512 // 16  # 32

        patch_pos_2d = patch_pos.reshape(1, grid_old, grid_old, dim).permute(0, 3, 1, 2)
        patch_pos_interpolated = F.interpolate(
            patch_pos_2d,
            size=(grid_new, grid_new),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_new = patch_pos_interpolated.permute(0, 2, 3, 1).reshape(1, grid_new * grid_new, dim)
        new_pos_embed = torch.cat([cls_pos, patch_pos_new], dim=1)  # [1, 1025, 768]

        self.vit_backbone.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        self.vit_backbone.image_size = 512

    def forward_vit_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1024 patch tokens from ViT at 512x512."""
        # Patch projection
        x_patches = self.vit_backbone.conv_proj(x)  # [B, 768, 32, 32]
        x_patches = x_patches.flatten(2).transpose(1, 2)  # [B, 1024, 768]

        # Class token
        cls_token = self.vit_backbone.class_token.expand(x.shape[0], -1, -1)  # [B, 1, 768]
        seq = torch.cat([cls_token, x_patches], dim=1)  # [B, 1025, 768]

        # Add positional embedding
        seq = seq + self.vit_backbone.encoder.pos_embedding
        seq = self.vit_backbone.encoder.dropout(seq)
        seq = self.vit_backbone.encoder.layers(seq)
        seq = self.vit_backbone.encoder.ln(seq)

        # Discard CLS token and return the 1024 spatial patch tokens
        return seq[:, 1:, :]  # [B, 1024, 768]

    def forward(
        self, x: torch.Tensor, return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        # 1. CNN Feature Maps
        cnn_map = self.cnn_backbone(x)  # [B, 1280, 16, 16]
        B, C, H, W = cnn_map.shape
        cnn_tokens = cnn_map.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 256, 1280]
        cnn_proj = self.cnn_proj(cnn_tokens)  # [B, 256, 512]

        # 2. ViT Tokens
        vit_tokens = self.forward_vit_tokens(x)  # [B, 1024, 768]
        vit_proj = self.vit_proj(vit_tokens)  # [B, 1024, 512]

        # 3. Bidirectional Cross-Attention
        fused_tokens, c2v_att, v2c_att = self.cross_attention(cnn_proj, vit_proj)  # [B, 1280, 512]

        # 4. Disease-Conditioned Attention
        disease_feat, disease_att = self.disease_attention(fused_tokens)  # [B, 6, 512]

        # 5. Disease Relationship Modeling (GCN)
        enhanced_feat, adj_matrix = self.gcn_relationship(disease_feat)  # [B, 6, 256], [B, 6, 6]

        # 6. Classification Logits
        logits = self.heads(enhanced_feat)  # [B, 6]

        output = {"logits": logits}
        if return_intermediates:
            output.update({
                "cnn_map": cnn_map,
                "fused_tokens": fused_tokens,
                "disease_feat": disease_feat,
                "disease_att": disease_att,
                "relationship_matrix": adj_matrix,
            })
        return output

    def freeze_backbones(self):
        """Freezes both CNN and ViT backbones for Stage 1 warmup."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
        self.cnn_backbone.eval()
        self.vit_backbone.eval()

    def unfreeze_backbones(self):
        """Unfreezes all parameters for Stage 2 end-to-end fine-tuning."""
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True
        for param in self.vit_backbone.parameters():
            param.requires_grad = True
        self.cnn_backbone.train()
        self.vit_backbone.train()

    @torch.no_grad()
    def predict_with_mc_dropout(
        self, x: torch.Tensor, num_samples: int = 15
    ) -> Dict[str, torch.Tensor]:
        """
        Executes Monte Carlo (MC) Dropout inference:
        Keeps dropout active during evaluation to sample T stochastic predictions.
        Computes predictive mean probabilities, epistemic variance, and normalized Shannon entropy.
        """
        self.eval()
        # Enable only dropout layers
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        sampled_probs = []
        for _ in range(num_samples):
            logits = self.forward(x)["logits"]
            probs = torch.softmax(logits, dim=-1)  # Multi-class categorical probability
            sampled_probs.append(probs.unsqueeze(0))

        # [T, B, 6]
        all_probs = torch.cat(sampled_probs, dim=0)

        # Predictive mean probability: [B, 6]
        mean_probs = all_probs.mean(dim=0)

        # Predictive epistemic variance (uncertainty): [B, 6]
        epistemic_var = all_probs.var(dim=0)

        # Normalized categorical Shannon entropy: [B] in [0, 1]
        eps = 1e-7
        p = torch.clamp(mean_probs, eps, 1.0)
        entropy = -torch.sum(p * torch.log2(p), dim=-1) / math.log2(NUM_CLASSES)
        mean_uncertainty = entropy

        # Confidence score (max predicted probability): [B]
        confidence = mean_probs.max(dim=1)[0]

        # Top prediction
        predicted_classes = torch.argmax(mean_probs, dim=1)

        return {
            "mean_probabilities": mean_probs,
            "epistemic_variance": epistemic_var,
            "uncertainty_entropy": mean_uncertainty,
            "confidence_score": confidence,
            "predicted_class": predicted_classes,
        }
