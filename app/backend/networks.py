"""
RESPIRA - Self-contained fusion network definitions.

IMPORTANT: These are faithful copies of the model classes in
src/fusion/*.py. They are defined here so the app NEVER imports
the script-style source modules (several of which execute
training / dataset loading at module import time, which would
re-train models and hang or corrupt checkpoints on startup).

Only the *architecture* is replicated; no weights are created
or saved here. The app loads the trained checkpoints into these
classes directly.
"""

import torch
import torch.nn as nn


# ============================================================
# FEATURE PROJECTION
# ============================================================


class FeatureProjection(nn.Module):
    """
    Projects EfficientNet (1280 -> D) and ViT (768 -> D)
    features into a common embedding dimension.
    """

    def __init__(
        self,
        efficientnet_dim=1280,
        vit_dim=768,
        projection_dim=512,
    ):
        super().__init__()

        self.efficientnet_projection = nn.Sequential(
            nn.Linear(efficientnet_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        self.vit_projection = nn.Sequential(
            nn.Linear(vit_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, efficientnet_tokens, vit_tokens):
        efficientnet_projected = self.efficientnet_projection(
            efficientnet_tokens
        )
        vit_projected = self.vit_projection(vit_tokens)
        return efficientnet_projected, vit_projected


# ============================================================
# CROSS-ATTENTION BLOCK
# ============================================================


class CrossAttentionBlock(nn.Module):
    """
    Query attends to another modality's tokens.
    Output retains the query modality's token count.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens, context_tokens):
        attended, attention_weights = self.attention(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=True,
        )

        x = self.norm1(query_tokens + attended)

        ff = self.feed_forward(x)

        x = self.norm2(x + ff)

        return x, attention_weights


# ============================================================
# BIDIRECTIONAL CROSS-ATTENTION
# ============================================================


class BidirectionalCrossAttention(nn.Module):
    """
    Direction 1: EfficientNet queries -> ViT context.
    Direction 2: ViT queries -> EfficientNet context.
    """

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.cnn_to_vit = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.vit_to_cnn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, cnn_tokens, vit_tokens):
        cnn_enhanced, cnn_to_vit_attention = self.cnn_to_vit(
            query_tokens=cnn_tokens,
            context_tokens=vit_tokens,
        )
        vit_enhanced, vit_to_cnn_attention = self.vit_to_cnn(
            query_tokens=vit_tokens,
            context_tokens=cnn_tokens,
        )

        return (
            cnn_enhanced,
            vit_enhanced,
            cnn_to_vit_attention,
            vit_to_cnn_attention,
        )


# ============================================================
# DISEASE-CONDITIONED ATTENTION
# ============================================================


class DiseaseConditionedAttention(nn.Module):
    """
    A learnable query per disease attends to the fused
    CNN + ViT token representation.

    Input:  [B, 245, 512]
    Output: [B, 6, 512], [B, 6, 245]
    """

    def __init__(
        self,
        num_diseases=6,
        embed_dim=512,
        num_heads=8,
    ):
        super().__init__()

        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        self.disease_queries = nn.Parameter(
            torch.randn(num_diseases, embed_dim) * 0.02
        )

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, fused_tokens):
        batch_size = fused_tokens.size(0)

        queries = self.disease_queries.unsqueeze(0)
        queries = queries.expand(batch_size, -1, -1)

        q = self.query_projection(queries)
        k = self.key_projection(fused_tokens)
        v = self.value_projection(fused_tokens)

        attended, attention_weights = self.attention(
            q,
            k,
            v,
            need_weights=True,
            average_attn_weights=True,
        )

        attended = self.norm(attended + queries)

        refined = self.ffn(attended)
        refined = self.ffn_norm(refined + attended)

        return refined, attention_weights


# ============================================================
# ADAPTIVE DISEASE-SPECIFIC FUSION
# ============================================================


class AdaptiveDiseaseFusion(nn.Module):
    """
    Input:  [B, 6, 512]
    Outputs disease_weights [B, 6], fused_features [B, 512],
    and logits [B, 6].
    """

    def __init__(
        self,
        feature_dim=512,
        num_classes=6,
        hidden_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.weight_network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.feature_transform = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # ----------------------------------------------------
        # Disease importance scores
        # ----------------------------------------------------

        disease_scores = self.weight_network(x)

        disease_weights = torch.softmax(
            disease_scores,
            dim=1,
        )

        # ----------------------------------------------------
        # Feature transformation per disease
        # ----------------------------------------------------

        transformed = self.feature_transform(x)

        # ----------------------------------------------------
        # Adaptive weighted fusion
        # ----------------------------------------------------

        fused_features = (
            transformed * disease_weights.unsqueeze(-1)
        ).sum(dim=1)

        # ----------------------------------------------------
        # Final logits
        # ----------------------------------------------------

        logits = self.classifier(fused_features)

        return {
            "fused_features": fused_features,
            "disease_weights": disease_weights,
            "logits": logits,
        }


# ============================================================
# DISEASE RELATIONSHIP MODELING
# ============================================================


class DiseaseRelationshipModel(nn.Module):
    """
    Multi-head self-attention across the six disease
    representations.

    Input:  [B, 6, 512]
    Output: enhanced [B, 6, 512], attention [B, heads, 6, 6]
    """

    def __init__(
        self,
        feature_dim=512,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        attended, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )

        x = self.norm1(x + attended)

        ff = self.feed_forward(x)

        x = self.norm2(x + ff)

        return x, attention_weights


# ============================================================
# MULTI-LABEL CLASSIFICATION HEADS
# ============================================================


class MultiLabelClassificationHeads(nn.Module):
    """
    Disease-specific classification heads.

    Inputs: disease_features [B, 6, 512] and
            relationship matrix [B, 6, 6].

    Each disease representation receives information from the
    others via the relationship matrix.

    Final output: logits [B, 6]
    """

    def __init__(
        self,
        num_classes=6,
        feature_dim=512,
        hidden_dim=256,
        dropout=0.3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.disease_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classification_heads = nn.ModuleList()

        for _ in range(num_classes):
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.classification_heads.append(head)

    def forward(self, disease_features, relationship):
        projected = self.disease_projection(disease_features)

        # Relationship propagation:
        # [B, 6, 6] x [B, 6, 256] -> [B, 6, 256]
        relational_features = torch.bmm(
            relationship,
            projected,
        )

        enhanced = projected + relational_features

        logits = []
        for disease_index in range(self.num_classes):
            disease_feature = enhanced[:, disease_index, :]
            disease_logit = self.classification_heads[
                disease_index
            ](disease_feature)
            logits.append(disease_logit)

        logits = torch.cat(logits, dim=1)

        return logits