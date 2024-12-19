# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 11:32
# @Author  : Biao
# @File    : model.py

"""
Model Architecture File

To Predict the Environmental Impact of a product

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
    """
    System Boundary Model to predict the classification of system boundary of target product in the life cycle assessment methodology
    """

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
    """
    Residual Block for the main branch of the model
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(ResBlock, self).__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class TextFeatureMainBranch(nn.Module):
    """
    Main Branch of the model to extract features from the text inputs
    
    Args:
        text_dim (int): Dimension of input text embeddings (default: 768)
        num_text_features (int): Number of text features to process (default: 6)
        hidden_dim (int): Dimension of hidden layers (default: 256)
        num_blocks (int): Number of residual blocks (default: 3)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)
        num_classes (int): Number of output classes (default: 6)
        use_residual (bool): Whether to use residual connections (default: True)
        use_attention (bool): Whether to use multi-head attention (default: True)
    """

    def __init__(self, text_dim=768, num_text_features=6, hidden_dim=256, num_blocks=3, 
                 num_heads=4, dropout=0.1, num_classes=6, use_residual=True, use_attention=True):
        super().__init__()

        self.use_residual = use_residual
        self.use_attention = use_attention

        # Project and compress input features
        self.input_projection = nn.Sequential(
            nn.Linear(text_dim * num_text_features, hidden_dim * 2),  # First expand dimensions
            nn.LayerNorm(hidden_dim * 2),                            # Normalize
            nn.GELU(),                                               # Non-linear activation
            nn.Dropout(dropout),                                     # Prevent overfitting
            nn.Linear(hidden_dim * 2, hidden_dim),                  # Compress to hidden dimension
        )

        # Optional residual blocks for better feature extraction
        if use_residual:
            self.res_blocks = nn.ModuleList(
                [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
            )

        # Optional multi-head attention layers for capturing relationships
        if use_attention:
            self.attention = nn.ModuleList(
                [nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout) 
                 for _ in range(2)]
            )

        # MLP with residual connection for final feature processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),    # Expand features
            nn.GELU(),                                # Non-linear activation
            nn.Dropout(dropout),                      # Regularization
            nn.Linear(hidden_dim * 4, hidden_dim * 2),# Gradual dimension reduction
            nn.GELU(),                               
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),    # Final compression
        )

        # Layer normalization for stabilizing training
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

    def forward(self, x):
        # Reshape input to batch_size x (text_dim * num_features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Apply residual blocks if enabled
        if self.use_residual:
            for res_block in self.res_blocks:
                x = x + 0.1 * res_block(x)  # Scale residual connections for stability
        
        # Apply attention if enabled
        if self.use_attention:
            x = x.unsqueeze(0)  # Add sequence length dimension for attention
            for attn_layer in self.attention:
                attn_out, _ = attn_layer(x, x, x)  # Self-attention (query=key=value)
                x = x + attn_out  # Residual connection
            x = x.squeeze(0)  # Remove sequence length dimension
            x = self.layer_norms[0](x)  # Normalize after attention
        
        # Final MLP with residual connection
        x = x + self.mlp(x)  # Residual connection
        x = self.layer_norms[1](x)  # Final normalization

        return x


class CommonTextFeatureModel(nn.Module):
    """
    A comprehensive model for processing text features with optional system boundary classification
    
    This model combines text feature processing with system boundary classification to predict 
    environmental impact. It supports various architectural choices including attention mechanisms
    and residual connections.

    Args:
        text_dim (int): Dimension of input text embeddings (default: 768)
        num_text_features (int): Number of text features to process (default: 6)
        hidden_dim (int): Dimension of hidden layers (default: 256)
        num_blocks (int): Number of residual blocks in main branch (default: 3)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate for regularization (default: 0.1)
        num_classes (int): Number of system boundary classes (default: 6)
        model_name (str): Name identifier for the model (default: "common_text_feature_model")
        model_path (str): Path to pretrained weights (default: None)
        use_attention (bool): Whether to use attention mechanism (default: True)
        use_system_boundary (bool): Whether to include system boundary classification (default: True)
        use_residual (bool): Whether to use residual connections (default: True)
    """

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

        # Store all configuration parameters for later reference and model reconstruction
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

        # Initialize the main feature extraction branch
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

        # Initialize system boundary classifier if enabled
        if use_system_boundary:
            self.classifier = SystemBoundaryModel(num_classes)

        output_layer_input_dim = hidden_dim

        # Configure output layers based on whether system boundary is used
        if use_system_boundary:
            # Output layers with concatenated system boundary features
            self.output_layer = nn.Sequential(
                nn.Linear(output_layer_input_dim + num_classes, hidden_dim),  # Combined features
                nn.LayerNorm(hidden_dim),                                     # Normalize
                nn.GELU(),                                                    # Non-linear activation
                nn.Dropout(dropout),                                          # Regularization
                nn.Linear(hidden_dim, 1),                                    # Final prediction
                nn.Softplus(),                                               # Ensure positive output
            )
        else:
            # Output layers without system boundary features
            self.output_layer = nn.Sequential(
                nn.Linear(output_layer_input_dim, hidden_dim),               # Process main features
                nn.LayerNorm(hidden_dim),                                    # Normalize
                nn.GELU(),                                                   # Non-linear activation
                nn.Dropout(dropout),                                         # Regularization
                nn.Linear(hidden_dim, 1),                                   # Final prediction
                nn.Softplus(),                                              # Ensure positive output
            )

        # Set up device for model computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained weights if provided
        if model_path:
            self.load_from_path(model_path)

    def forward(self, text_inputs_embeddings, system_boundary_embeddings, *args):
        """
        Forward pass of the model
        
        Args:
            text_inputs_embeddings: Tensor of text embeddings
            system_boundary_embeddings: Tensor of system boundary embeddings
            *args: Additional arguments (not used)
            
        Returns:
            Tensor: Final prediction scores
        """
        # Extract features from the main branch
        main_features = self.main_branch(text_inputs_embeddings)

        # Process system boundary if enabled
        if self.config["use_system_boundary"]:
            system_boundary = self.classifier(system_boundary_embeddings)
            # Concatenate main features with system boundary predictions
            final_features = torch.cat([main_features, system_boundary], dim=1)
        else:
            final_features = main_features

        # Generate final predictions
        return self.output_layer(final_features).squeeze(-1)

    def forward_till_fusion(
        self, text_inputs_embeddings, system_boundary_embeddings, *args
    ):
        """
        Forward pass that returns intermediate features before final prediction
        
        This method is useful for feature visualization or analysis
        
        Args:
            text_inputs_embeddings: Tensor of text embeddings
            system_boundary_embeddings: Tensor of system boundary embeddings
            *args: Additional arguments (not used)
            
        Returns:
            Tensor: Combined features before final prediction layer
        """
        main_features = self.main_branch(text_inputs_embeddings)

        if self.config["use_system_boundary"]:
            system_boundary = self.classifier(system_boundary_embeddings)
            final_features = torch.cat([main_features, system_boundary], dim=1)
        else:
            final_features = main_features

        return final_features

    @property
    def model_config(self):
        """
        Property to access model configuration
        
        Returns:
            dict: Model configuration parameters
        """
        return self.config

    def load_from_path(self, model_path):
        """
        Load pretrained weights from file
        
        Args:
            model_path (str): Path to the saved model weights
        """
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        self.to(self.device)
        logger.info(f"Loaded model from {model_path}")


MODELS = {
    "common_text_feature_model": CommonTextFeatureModel,
}
