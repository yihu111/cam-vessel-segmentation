"""
PyTorch Lightning models for medical image segmentation.
"""

import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

# Import necessary utilities
from ..models.smp_models import create_model


class SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for medical image segmentation.
    Uses a combination of Dice and Focal losses for vessel segmentation.
    """
    
    def __init__(
        self,
        model_name: str = 'unetplusplus',
        encoder_name: str = 'resnet34',
        in_channels: int = 1,
        classes: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        threshold: float = 0.5,
        pretrained_weights: Optional[str] = None,
        freeze_encoder: bool = False,
        custom_backbone: Optional[torch.nn.Module] = None,
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model_name: Name of the segmentation model architecture ('unet', 'unetplusplus', etc.)
            encoder_name: Name of the encoder backbone ('resnet34', 'efficientnet-b0', etc.)
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            classes: Number of output classes (1 for binary segmentation)
            learning_rate: Initial learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            threshold: Threshold for binary prediction
            pretrained_weights: Path to pretrained weights (optional)
            freeze_encoder: Whether to freeze the encoder for transfer learning
        """
        super().__init__()
        self.save_hyperparameters()
        # Use custom backbone if provided (for hybrid/foundation models)
        if custom_backbone is not None:
            self.model = custom_backbone
        else:
            # Create model
            self.model = create_model(
                model_name=model_name,
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channels,
                classes=classes
            )
            # Load pretrained weights if provided (skip if loading from checkpoint during inference)
            if pretrained_weights is not None and os.path.exists(pretrained_weights):
                checkpoint = torch.load(pretrained_weights, map_location='cpu')
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    # Our custom format
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded pretrained weights from {pretrained_weights} (using model_state_dict)")
                elif 'state_dict' in checkpoint:
                    # PyTorch Lightning format
                    # Extract state dict and remove 'model.' prefix
                    state_dict = {}
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('model.'):
                            # Remove 'model.' prefix
                            new_key = key[6:]  # Skip 'model.'
                            state_dict[new_key] = value
                        else:
                            # Keep original key for other components
                            state_dict[key] = value
                    # Load the modified state dict
                    self.model.load_state_dict(state_dict)
                    print(f"Loaded pretrained weights from {pretrained_weights} (using state_dict with prefix handling)")
                else:
                    print(f"WARNING: Checkpoint format not recognized. Available keys: {list(checkpoint.keys())}")
                    print("Continuing without loading pretrained weights.")
                # Option to freeze encoder (useful for transfer learning)
                freeze_encoder = self.hparams.get('freeze_encoder', False)
                if freeze_encoder and hasattr(self.model, 'encoder'):
                    print("Freezing encoder for transfer learning")
                    for param in self.model.encoder.parameters():
                        param.requires_grad = False
        
        # Initialize loss functions
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        
        # Set loss weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Set threshold for binary prediction
        self.threshold = threshold
        
        # Print model information
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {model_name} with {encoder_name}")
        print(f"Parameters: {num_params:,}")
        print(f"Loss: Combined ({dice_weight:.1f}*Dice + {focal_weight:.1f}*Focal)")
        
        # We'll use custom metric implementations instead of torchmetrics
        # This gives us complete control over metric calculation
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def _compute_loss(self, outputs, targets):
        """
        Compute the combined loss for segmentation.
        
        Args:
            outputs: Model predictions
            targets: Ground truth masks
            
        Returns:
            loss: Computed loss value
        """
        # Compute individual loss components
        dice = self.dice_loss(outputs, targets)
        focal = self.focal_loss(outputs, targets)
        
        # Combine losses with weights
        loss = self.dice_weight * dice + self.focal_weight * focal
        
        return loss
    
    def _compute_metrics(self, preds, targets):
        """
        Compute custom metrics for segmentation evaluation.
        
        Args:
            preds: Model predictions after thresholding (binary)
            targets: Ground truth masks (binary)
            
        Returns:
            dict: Dictionary of computed metrics
        """
        # Ensure inputs are binary
        preds = (preds > self.threshold).float() if not torch.all((preds == 0) | (preds == 1)) else preds
        targets = (targets > self.threshold).float() if not torch.all((targets == 0) | (targets == 1)) else targets
        
        # Move to CPU for metric calculation if needed
        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        
        # Calculate Dice coefficient (F1 score)
        # Formula: 2*|X∩Y|/(|X|+|Y|)
        smooth = 1e-7  # Smoothing factor to avoid division by zero
        intersection = torch.sum(preds_cpu * targets_cpu)
        dice = (2. * intersection + smooth) / (torch.sum(preds_cpu) + torch.sum(targets_cpu) + smooth)
        
        # Calculate precision: TP / (TP + FP)
        true_positives = torch.sum(preds_cpu * targets_cpu)
        predicted_positives = torch.sum(preds_cpu)
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        
        # Calculate recall: TP / (TP + FN)
        actual_positives = torch.sum(targets_cpu)
        recall = (true_positives + smooth) / (actual_positives + smooth)
        
        # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
        total_pixels = torch.numel(preds_cpu)
        true_negatives = total_pixels - (predicted_positives + actual_positives - true_positives)
        accuracy = (true_positives + true_negatives) / total_pixels
        
        return {
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Mini-batch from dataloader
            batch_idx: Batch index
            
        Returns:
            loss: Computed loss value
        """
        # Unpack batch - handle both dictionary and tuple formats
        if isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        elif len(batch) == 3:
            images, masks, _ = batch
        elif len(batch) == 2:
            images, masks = batch
        else:
            raise ValueError(f"Unexpected batch format: {batch}")
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self._compute_loss(outputs, masks)
        
        # Apply threshold to get binary predictions
        preds = (outputs > self.threshold).float()
        
        # Calculate custom metrics
        metrics = self._compute_metrics(preds, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Mini-batch from dataloader
            batch_idx: Batch index
            
        Returns:
            loss: Computed loss value
        """
        # Unpack batch - handle both dictionary and tuple formats
        if isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
            original_images = batch.get('original_image', images)
        elif len(batch) == 3:
            images, masks, _ = batch
            original_images = images
        elif len(batch) == 2:
            images, masks = batch
            original_images = images
        else:
            raise ValueError(f"Unexpected batch format: {batch}")
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self._compute_loss(outputs, masks)
        
        # Apply threshold to get binary predictions
        preds = (outputs > self.threshold).float()
        
        # Calculate custom metrics
        metrics = self._compute_metrics(preds, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        # For visualization in validation_epoch_end - store ORIGINAL images
        if batch_idx == 0:
            self._sample_imgs = original_images
            self._sample_masks = masks
            self._sample_outputs = outputs
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Mini-batch from dataloader
            batch_idx: Batch index
        """
        # Unpack batch - handle both dictionary and tuple formats
        if isinstance(batch, dict):
            images = batch['image']
            masks = batch['mask']
        elif len(batch) == 3:
            images, masks, _ = batch
        elif len(batch) == 2:
            images, masks = batch
        else:
            raise ValueError(f"Unexpected batch format: {batch}")
        
        # Forward pass
        outputs = self(images)
        
        # Calculate loss
        loss = self._compute_loss(outputs, masks)
        
        # Apply threshold to get binary predictions
        preds = (outputs > self.threshold).float()
        
        # Calculate custom metrics
        metrics = self._compute_metrics(preds, masks)
        
        # Log metrics
        # self.log('test_loss', loss, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log_dict({f'test_{k}': v for k, v in metrics.items()}, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        (Updated from validation_epoch_end to comply with Lightning 2.0)
        """
        # Create visualization figure
        if hasattr(self, '_sample_imgs') and self.logger is not None:
            # Take only the first few samples
            n_samples = min(4, self._sample_imgs.size(0))
            
            # Create figure
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            
            # Handle different axis shapes properly
            if n_samples == 1:
                # When there's only one sample, axes isn't a 2D array but a 1D array
                axes = [axes]  # Wrap in list for consistent indexing below
            
            # Plot images, masks, and predictions
            for i in range(n_samples):
                # Get data
                img = self._sample_imgs[i].cpu().detach().numpy()
                mask = self._sample_masks[i].cpu().detach().numpy()
                pred = (self._sample_outputs[i] > self.threshold).cpu().detach().numpy()
                
                # For grayscale images (1 channel)
                if img.shape[0] == 1:
                    img = img[0]
                    mask = mask[0]
                    pred = pred[0]
                else:
                    # For RGB images (3 channels), convert to numpy format (H,W,C)
                    img = np.transpose(img, (1, 2, 0))
                
                # Handle different axes shapes properly
                if n_samples == 1:
                    # Plot original image
                    axes[0][0].imshow(img, cmap='gray')
                    axes[0][0].set_title('Image')
                    axes[0][0].axis('off')
                    
                    # Plot ground truth mask
                    axes[0][1].imshow(mask.squeeze(), cmap='gray')
                    axes[0][1].set_title('Ground Truth')
                    axes[0][1].axis('off')
                    
                    # Plot prediction
                    axes[0][2].imshow(pred.squeeze(), cmap='gray')
                    axes[0][2].set_title('Prediction')
                    axes[0][2].axis('off')
                else:
                    # Plot original image
                    axes[i, 0].imshow(img, cmap='gray')
                    axes[i, 0].set_title('Image')
                    axes[i, 0].axis('off')
                    
                    # Plot ground truth mask
                    axes[i, 1].imshow(mask.squeeze(), cmap='gray')
                    axes[i, 1].set_title('Ground Truth')
                    axes[i, 1].axis('off')
                    
                    # Plot prediction
                    axes[i, 2].imshow(pred.squeeze(), cmap='gray')
                    axes[i, 2].set_title('Prediction')
                    axes[i, 2].axis('off')
            
            # Tight layout
            plt.tight_layout()
            
            # Log figure to logger
            if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                self.logger.experiment.add_figure(
                    'predictions',
                    fig,
                    global_step=self.current_epoch
                )
            
            # Close figure to free memory
            plt.close(fig)
            
            # Tight layout
            plt.tight_layout()
            
            # Log figure to logger
            if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                self.logger.experiment.add_figure(
                    'predictions',
                    fig,
                    global_step=self.current_epoch
                )
            
            # Close figure to free memory
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay)
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time (10, 20, 40...)
            eta_min=1e-7  # Minimum learning rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }