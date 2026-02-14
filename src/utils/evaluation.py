"""
Evaluation utilities for segmentation models.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_dir: Optional[str] = None,
    save_predictions: bool = False,
    num_samples_to_save: int = 10,
    return_predictions: bool = False,
    metrics_to_compute: List[str] = None,
    set_name: str = "Test Set"
) -> Dict[str, float]:
    """
    Evaluate a segmentation model on the test set.
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for the test set
        device: Device to run evaluation on
        threshold: Threshold for binary prediction (default: 0.5)
        save_dir: Directory to save visualizations (None = don't save)
        save_predictions: Whether to save prediction visualizations
        num_samples_to_save: Number of sample predictions to save
        return_predictions: Whether to return raw predictions
        metrics_to_compute: List of metrics to compute ['accuracy', 'precision', 'recall', 'dice', 'iou', 'specificity']
        set_name: Name of the dataset being evaluated (for display purposes, default: "Test Set")
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Set default metrics if None provided
    if metrics_to_compute is None:
        metrics_to_compute = ['dice', 'iou', 'precision', 'recall', 'accuracy', 'specificity']
    
    # Convert to lowercase for consistency
    metrics_to_compute = [m.lower() for m in metrics_to_compute]
    
    # Initialize metrics
    dice_scores = [] if 'dice' in metrics_to_compute else None
    iou_scores = [] if 'iou' in metrics_to_compute else None
    precision_scores = [] if 'precision' in metrics_to_compute else None
    recall_scores = [] if 'recall' in metrics_to_compute else None
    accuracy_scores = [] if 'accuracy' in metrics_to_compute else None
    specificity_scores = [] if 'specificity' in metrics_to_compute else None
    
    # Initialize prediction tracking if requested
    all_images = []
    all_masks = []
    all_preds = []
    all_probs = []
    all_filenames = []
    
    # Create save directory if needed
    if save_dir and save_predictions:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Use tqdm for progress bar
        pbar = tqdm(test_loader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch - handle both dictionary and tuple formats
            if isinstance(batch, dict):
                images = batch['image']
                masks = batch['mask']
                filenames = batch['filename']
                original_images = batch.get('original_image', images)
            else:
                # Fallback for tuple format
                images, masks, filenames = batch
                original_images = images
            
            # Move to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
            # Move to CPU for metric calculation
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            # Ensure masks are binary (0 or 1)
            masks_np = (masks_np > 0.5).astype(np.float32)
            
            # Save some predictions for visualization if requested
            if (save_predictions or return_predictions) and batch_idx < num_samples_to_save:
                all_images.extend(original_images.cpu())
                all_masks.extend(masks.cpu())
                all_preds.extend(preds.cpu())
                all_probs.extend(probs.cpu())
                all_filenames.extend(filenames)
            
            # Calculate metrics for each image in batch
            for i in range(preds_np.shape[0]):
                pred_flat = preds_np[i].flatten()
                mask_flat = masks_np[i].flatten()
                
                # Calculate Dice score (F1)
                if dice_scores is not None:
                    dice = f1_score(mask_flat, pred_flat, zero_division=1)
                    dice_scores.append(dice)
                
                # Calculate IoU (Jaccard)
                if iou_scores is not None:
                    iou = jaccard_score(mask_flat, pred_flat, zero_division=1)
                    iou_scores.append(iou)
                
                # Calculate precision
                if precision_scores is not None:
                    precision = precision_score(mask_flat, pred_flat, zero_division=1)
                    precision_scores.append(precision)
                
                # Calculate recall (sensitivity)
                if recall_scores is not None:
                    recall = recall_score(mask_flat, pred_flat, zero_division=1)
                    recall_scores.append(recall)
                
                # Calculate accuracy
                if accuracy_scores is not None:
                    accuracy = accuracy_score(mask_flat, pred_flat)
                    accuracy_scores.append(accuracy)
                
                # Calculate specificity (true negative rate)
                if specificity_scores is not None:
                    tn = np.sum((1 - mask_flat) * (1 - pred_flat))
                    fp = np.sum((1 - mask_flat) * pred_flat)
                    specificity = tn / (tn + fp + 1e-8)
                    specificity_scores.append(specificity)
            
            # Update progress bar with current metrics
            postfix_dict = {}
            if dice_scores is not None:
                postfix_dict['Dice'] = f'{np.mean(dice_scores):.4f}'
            if accuracy_scores is not None:
                postfix_dict['Acc'] = f'{np.mean(accuracy_scores):.4f}'
            pbar.set_postfix(postfix_dict)
    
    # Calculate average metrics
    metrics = {}
    if dice_scores is not None:
        metrics['dice'] = np.mean(dice_scores)
    if iou_scores is not None:
        metrics['iou'] = np.mean(iou_scores)
    if precision_scores is not None:
        metrics['precision'] = np.mean(precision_scores)
    if recall_scores is not None:
        metrics['recall'] = np.mean(recall_scores)
    if accuracy_scores is not None:
        metrics['accuracy'] = np.mean(accuracy_scores)
    if specificity_scores is not None:
        metrics['specificity'] = np.mean(specificity_scores)
    
    # Print detailed metrics
    print(f"\n{set_name} Metrics:")
    if 'dice' in metrics:
        print(f"Dice Score: {metrics['dice']:.4f}")
    if 'iou' in metrics:
        print(f"IoU Score: {metrics['iou']:.4f}")
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}")
    if 'recall' in metrics:
        print(f"Recall/Sensitivity: {metrics['recall']:.4f}")
    if 'specificity' in metrics:
        print(f"Specificity: {metrics['specificity']:.4f}")
    if 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Save visualizations if requested
    if save_predictions and save_dir and len(all_images) > 0:
        save_prediction_visualizations(all_images, all_masks, all_preds, all_filenames, save_dir, probabilities=all_probs)
    
    # Return raw predictions if requested
    if return_predictions:
        metrics['images'] = all_images
        metrics['masks'] = all_masks
        metrics['predictions'] = all_preds
        metrics['filenames'] = all_filenames
    
    return metrics


def save_prediction_visualizations(
    images: List[torch.Tensor],
    masks: List[torch.Tensor],
    predictions: List[torch.Tensor],
    filenames: List[str],
    save_dir: str,
    probabilities: List[torch.Tensor] = None
) -> None:
    """
    Save visualizations of model predictions.
    
    Args:
        images: List of input images (torch tensors)
        masks: List of ground truth masks (torch tensors)
        predictions: List of predicted masks (torch tensors)
        filenames: List of filenames for the images
        save_dir: Directory to save visualizations
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Use non-interactive backend to avoid Qt issues
    plt.switch_backend('Agg')
    
    for i, (image, mask, pred, prob, filename) in enumerate(zip(images, masks, predictions, probabilities, filenames)):
        # Convert to numpy and move channel dimension to end
        if image.shape[0] == 1:  # Grayscale
            img_np = image.squeeze().numpy()
        else:  # RGB
            img_np = image.permute(1, 2, 0).numpy()
        
        mask_np = mask.squeeze().numpy()
        prob_np = prob.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot original image
        if image.shape[0] == 1:  # Grayscale
            axes[0].imshow(img_np, cmap='gray')
        else:  # RGB
            axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Probability map
        axes[2].imshow(prob_np, cmap='viridis')
        axes[2].set_title('Probability Map')
        axes[2].axis('off')
        
        # Thresholded prediction
        axes[3].imshow(pred_np, cmap='gray')
        axes[3].set_title('Prediction')
        axes[3].axis('off')
        
        # Set figure title
        plt.suptitle(f'File: {filename}')
        plt.tight_layout()
        
        # Save figure
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        fig.savefig(os.path.join(save_dir, f'{base_filename}_prediction.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


def visualize_predictions(
    images: List[torch.Tensor],
    masks: List[torch.Tensor],
    predictions: List[torch.Tensor],
    filenames: List[str],
    num_samples: int = 5
) -> None:
    """
    Visualize model predictions in a Jupyter notebook.
    
    Args:
        images: List of input images (torch tensors)
        masks: List of ground truth masks (torch tensors)
        predictions: List of predicted masks (torch tensors)
        filenames: List of filenames for the images
        num_samples: Number of samples to visualize
    """
    # Limit the number of samples
    num_samples = min(num_samples, len(images))
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Ensure axes is 2D even for num_samples=1
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Set column titles
    axes[0, 0].set_title('Original Image')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].set_title('Prediction')
    
    for i in range(num_samples):
        image = images[i]
        mask = masks[i]
        pred = predictions[i]
        filename = filenames[i]
        
        # Convert to numpy and move channel dimension to end
        if image.shape[0] == 1:  # Grayscale
            img_np = image.squeeze().numpy()
        else:  # RGB
            img_np = image.permute(1, 2, 0).numpy()
            
        mask_np = mask.squeeze().numpy()
        pred_np = pred.squeeze().numpy()
        
        # Plot original image
        if image.shape[0] == 1:  # Grayscale
            axes[i, 0].imshow(img_np, cmap='gray')
        else:  # RGB
            axes[i, 0].imshow(img_np)
        axes[i, 0].axis('off')
        
        # Plot ground truth mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].axis('off')
        
        # Add filename as row title
        axes[i, 0].set_ylabel(os.path.basename(filename), fontsize=10, rotation=0, labelpad=40)
    
    plt.tight_layout()
    plt.show()
