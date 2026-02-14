"""
Segmentation model wrappers using segmentation_models_pytorch.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Dict, Any, List


def init_weights(m):
    """Initialize model weights to prevent NaN issues"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Use a more conservative initialization to prevent numerical instability
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            # Initialize biases to small positive values to help with initial activations
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def create_model(
    model_name: str = 'unet',
    encoder_name: str = 'resnet18',
    in_channels: int = 1,
    classes: int = 1,
    activation: Optional[str] = None,
    encoder_weights: Optional[str] = 'imagenet') -> torch.nn.Module:
    """
    Create a segmentation model using segmentation_models_pytorch.
    
    Args:
        model_name: Model architecture ('Unet', 'UnetPlusPlus', 'DeepLabV3', etc.)
        encoder_name: Encoder backbone ('resnet18', 'resnet34', 'efficientnet-b0', etc.)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        classes: Number of output classes (1 for binary segmentation)
        activation: Final activation function (None = no activation, 'sigmoid', etc.)
        encoder_weights: Pre-trained weights for encoder ('imagenet' or None)
    
    Returns:
        Initialized model
    """
    # Normalize model name for SMP
    model_name = model_name.lower()
    
    # Map model name to SMP class
    model_map = {
        'unet': smp.Unet,
        'unetplusplus': smp.UnetPlusPlus,
    }
    
    # Get the model class from the mapping
    model_class = model_map.get(model_name)
    
    if model_class is None:
        raise ValueError(f"Unknown model_name: {model_name}. Valid options are: {list(model_map.keys())}")
    
    # Create the model
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation
    )
    
    # Apply custom initialization for UNet++ to prevent NaN issues
    if model_name.lower() == 'unetplusplus':
        print("Using custom weight initialization for UNet++ to prevent NaN issues")
        # Initialize the decoder (but not the encoder if using pretrained weights)
        if hasattr(model, 'decoder'):
            model.decoder.apply(init_weights)
        if hasattr(model, 'segmentation_head'):
            model.segmentation_head.apply(init_weights)
    
    return model
