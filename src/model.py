# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 11:32
# @Author  : Biao
# @File    : model.py

"""
Model Architecture File

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import logger


def print_grad(grad):
    print("\nGradient:", grad)
    return grad


def clip_grad(grad):
    grad = torch.clamp(grad, -1, 1)
    return grad


class SystemBoundaryModel(nn.Module):
    def __init__(self, num_classes=6, text_dim=768):
        super(SystemBoundaryModel, self).__init__()

        self.system_boundary_classifier = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, system_boundary_embedding):
        logits = self.system_boundary_classifier(system_boundary_embedding)
        return F.softmax(logits, dim=1)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        # x = self.batch_norm(x)

        return x


class TextFeatureMainBranch(nn.Module):
    def __init__(
        self,
        text_dim=768,
        num_text_features=6,
        hidden_dim=256,
        num_blocks=3,
        num_heads=4,
        dropout=0.1,
        num_classes=6,
        use_residual=True,
        use_attention=True,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.use_attention = use_attention

        self.input_projection = nn.Sequential(
            nn.Linear(text_dim * num_text_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        if use_residual:
            self.res_blocks = nn.ModuleList(
                [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
            )

        if use_attention:
            self.attention = nn.ModuleList(
                [
                    nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
                    for _ in range(2)
                ]
            )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

    def forward(self, x):
        # Flatten and project
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.input_projection(x)

        # ResNet blocks with scaled residual
        if self.use_residual:
            for res_block in self.res_blocks:
                x = x + 0.1 * res_block(x)

        # Self-attention blocks
        if self.use_attention:
            x = x.unsqueeze(0)
            for attn_layer in self.attention:
                attn_out, _ = attn_layer(x, x, x)
                x = x + attn_out
            x = x.squeeze(0)
            x = self.layer_norms[0](x)

        # MLP with residual
        x = x + self.mlp(x)
        x = self.layer_norms[1](x)

        return x


class CommonTextFeatureModel(nn.Module):
    def __init__(
        self,
        text_dim=768,
        num_text_features=6,
        hidden_dim=256,
        num_blocks=3,
        num_heads=4,
        dropout=0.1,
        num_classes=6,
        model_name="common_text_feature_model",
        model_path=None,
        use_attention=True,
        use_system_boundary=True,
        use_residual=True,
    ):
        super().__init__()

        # Store config
        self.config = {
            "text_dim": text_dim,
            "num_text_features": num_text_features,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "num_heads": num_heads,
            "dropout": dropout,
            "num_classes": num_classes,
            "model_name": model_name,
            "model_path": model_path,
            "use_system_boundary": use_system_boundary,
            "use_attention": use_attention,
            "use_residual": use_residual,
        }
        self.use_attention = use_attention
        self.use_system_boundary = use_system_boundary

        # Main components
        self.main_branch = TextFeatureMainBranch(
            text_dim,
            num_text_features,
            hidden_dim,
            num_blocks,
            num_heads,
            dropout,
            use_residual=use_residual,
            use_attention=use_attention,
        )

        if use_system_boundary:
            self.classifier = SystemBoundaryModel(num_classes)

        output_layer_input_dim = hidden_dim

        if use_system_boundary:
            # Output layers
            self.output_layer = (
                nn.Sequential(
                    nn.Linear(output_layer_input_dim + num_classes, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus(),
                )
            )
        else:
            self.output_layer = (
                nn.Sequential(
                    nn.Linear(output_layer_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Softplus(),
                )
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained weights if path provided
        if model_path:
            self.load_from_path(model_path)

    def forward(self, text_inputs_embeddings, system_boundary_embeddings, *args):
        # Get features from both branches
        main_features = self.main_branch(text_inputs_embeddings)

        # Get system boundary predictions and combine
        if self.config["use_system_boundary"]:

            # Get system boundary predictions and combine
            system_boundary = self.classifier(system_boundary_embeddings)
            final_features = torch.cat([main_features, system_boundary], dim=1)
        else:
            final_features = main_features

        return self.output_layer(final_features).squeeze(-1)

    def forward_till_fusion(
        self, text_inputs_embeddings, system_boundary_embeddings, *args
    ):
        # Get features from both branches
        main_features = self.main_branch(text_inputs_embeddings)

        # Get system boundary predictions and combine
        if self.config["use_system_boundary"]:

            # Get system boundary predictions and combine
            system_boundary = self.classifier(system_boundary_embeddings)
            final_features = torch.cat([main_features, system_boundary], dim=1)
        else:
            final_features = main_features

        return final_features

    @property
    def model_config(self):
        return self.config

    def load_from_path(self, model_path):
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        self.to(self.device)
        logger.info(f"Loaded model from {model_path}")


MODELS = {
    "common_text_feature_model": CommonTextFeatureModel,
}
