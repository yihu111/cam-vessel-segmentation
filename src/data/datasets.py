"""
Dataset classes for medical image segmentation (fundus and CAM vessel images).
"""

import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Tuple, Optional
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random


class MedicalImageDataset(Dataset):
    """
    Unified dataset class for medical image segmentation.
    Can handle both retinal fundus images and CAM images with the same transformation pipeline.
    """
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str,
                 image_size: int = 1024,
                 crop_size: int = 512,
                 augmentation: bool = True,
                 gamma: float = 1.2,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 file_extension: str = None,
                 use_normalization: bool = False,
                 lower_percentile: float = 1.0,
                 upper_percentile: float = 99.0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.use_normalization = use_normalization
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
        # Initialize contrast enhancement
        self.contrast_enhancer = ContrastEnhancedTransform(
            gamma=gamma, 
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size
        )
        
        # Validate directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        # Allow filtering by file extension if specified
        if file_extension:
            self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(file_extension)])
        else:
            # Accept common image formats
            self.image_files = sorted([f for f in os.listdir(image_dir) 
                                     if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir} with specified extension(s)")
        
    def __len__(self):
        return len(self.image_files)
    
    def normalize(self, image):
        """
        Per-image, per-channel percentile normalization + z-score.
        
        Args:
            image: PIL Image with values in [0, 255]
        
        Returns:
            Normalized tensor with zero mean and unit variance per channel
        """
        if not self.use_normalization:
            # Just convert to [0, 1] tensor
            return TF.to_tensor(image)
        
        # Convert PIL to numpy [H, W, C] in range [0, 255]
        image_np = np.array(image).astype(np.float32)
        
        # Normalize each channel separately on RAW values
        normalized = np.zeros_like(image_np)
        
        for c in range(image_np.shape[2]):
            channel = image_np[:, :, c]
            
            # Compute percentiles on RAW 0-255 range
            lower = np.percentile(channel, self.lower_percentile)
            upper = np.percentile(channel, self.upper_percentile)
            
            # Clip to percentile range
            channel_clipped = np.clip(channel, lower, upper)
            
            # Z-score normalization
            mean = channel_clipped.mean()
            std = channel_clipped.std()
            
            if std > 1e-6:  # Avoid division by zero
                normalized[:, :, c] = (channel_clipped - mean) / std
            else:
                normalized[:, :, c] = channel_clipped - mean
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        
        return tensor
    
    def transform(self, image, mask):
        """
        Apply joint transformations to image and mask to ensure they stay aligned.
        Uses functional transforms for precise control.
        
        CRITICAL ORDER:
        1. Resize both image and mask to target size
        2. Apply contrast enhancement (on RAW pixel values)
        3. Apply other transformations (flipping, rotation, etc.)
        4. Convert mask to tensor
        """
        # 1. First, resize both image and mask to target size
        image = TF.resize(image, (self.image_size, self.image_size))
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.NEAREST)
        
        # 2. Apply contrast enhancement (after initial resize)
        image = self.contrast_enhancer(image)
        
        # 3. Apply additional augmentations if enabled
        if self.augmentation:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random rotation (0, 90, 180, or 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Random zoom (scale variation)
            if random.random() > 0.7:
                scale_factor = random.uniform(1.2, 2.0)
                new_size = int(self.image_size * scale_factor)
                
                # Resize to scaled size
                image = TF.resize(image, (new_size, new_size))
                mask = TF.resize(mask, (new_size, new_size), interpolation=TF.InterpolationMode.NEAREST)
                
                # Then crop or pad back to original size
                if scale_factor > 1.0:  # Zoomed in, need to crop
                    i = (new_size - self.image_size) // 2
                    j = (new_size - self.image_size) // 2
                    image = TF.crop(image, i, j, self.image_size, self.image_size)
                    mask = TF.crop(mask, i, j, self.image_size, self.image_size)
                else:  # Zoomed out, need to pad
                    padding = (self.image_size - new_size) // 2
                    image = TF.pad(image, padding)
                    mask = TF.pad(mask, padding)
            
            # Random elastic deformation (simulates tissue deformation)
            if random.random() > 0.7:
                image = self._elastic_transform(image, alpha=50, sigma=3)
                mask = self._elastic_transform(mask, alpha=50, sigma=3)
                
            # Apply random crop to crop_size (default 512x512) as part of augmentation
            # Get current dimensions
            width, height = image.size
            
            # Only crop if the image is larger than crop_size
            if width > self.crop_size and height > self.crop_size:
                # Get random crop coordinates
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                # Apply crop to both image and mask
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
        
        # 4. Convert mask to tensor (keep image as PIL with RAW 0-255 values)
        mask = TF.to_tensor(mask)
        
        # DO NOT normalize image here
        return image, mask
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Handle different file extensions between images and masks
        # Try different extensions for the mask file
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # If mask doesn't exist with the same extension, try common alternatives
        if not os.path.exists(mask_path):
            # Get filename without extension
            base_name = os.path.splitext(img_name)[0]
            
            # Try common mask extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path
                    break
        
        # Open images
        image = Image.open(img_path).convert('RGB')
        
        if os.path.exists(mask_path):
            # Convert to '1' mode (binary) for consistent binary mask representation
            mask = Image.open(mask_path).convert('1')
        else:
            # Create empty mask with 0 values
            print(f"Warning: Mask not found for {img_name}, creating empty mask")
            mask = Image.new('1', image.size, 0)
        
        # Store original RGB image for visualization (resize and normalize to [0, 1])
        original_image_rgb = image.resize((self.image_size, self.image_size))
        original_image_rgb = np.array(original_image_rgb).astype(np.float32) / 255.0
        
        # Convert to tensor
        original_tensor = torch.from_numpy(original_image_rgb).permute(2, 0, 1).float()
        
        # Apply joint transformations
        image_pil, mask_tensor = self.transform(image, mask)
        
        # Apply normalization on RAW pixel values (0-255)
        image_tensor = self.normalize(image_pil)
        
        # Return as dictionary for clarity
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'original_image': original_tensor,
            'filename': img_name
        }
    
    def _elastic_transform(self, image, alpha=50, sigma=5):
        """
        Apply elastic deformation to simulate tissue deformation.
        
        Args:
            image: PIL Image
            alpha: Deformation intensity
            sigma: Smoothness of deformation
        
        Returns:
            Transformed PIL Image
        """
        # Convert to numpy
        image_np = np.array(image)
        shape = image_np.shape
        
        # Generate random displacement fields
        dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (0, 0), sigma) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Add displacement
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        
        # Fix: Handle boolean masks (convert to uint8 for OpenCV)
        if image_np.dtype == bool:
            image_np = image_np.astype(np.uint8) * 255
            distorted = cv2.remap(image_np, indices[1], indices[0], 
                                 interpolation=cv2.INTER_NEAREST, 
                                 borderMode=cv2.BORDER_REFLECT)
            distorted = (distorted > 127).astype(np.uint8) * 255
            return Image.fromarray(distorted).convert('1')
        # Apply transformation
        elif len(shape) == 3:  # Color image
            distorted = cv2.remap(image_np, indices[1], indices[0], 
                                 interpolation=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_REFLECT)
        else:  # Grayscale
            distorted = cv2.remap(image_np, indices[1], indices[0], 
                                 interpolation=cv2.INTER_NEAREST, 
                                 borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(distorted)


# Define aliases for backward compatibility and clarity
class FundusDataset(MedicalImageDataset):
    """Alias for MedicalImageDataset optimized for fundus images"""
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str,
                 **kwargs):
        # Default to .png for fundus images
        kwargs.setdefault('file_extension', '.png')
        # Per-image normalization by default
        kwargs.setdefault('use_normalization', True)
        kwargs.setdefault('lower_percentile', 1.0)
        kwargs.setdefault('upper_percentile', 99.0)
        super().__init__(image_dir, mask_dir, **kwargs)


class CAMDataset(MedicalImageDataset):
    """Alias for MedicalImageDataset optimized for CAM images"""
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str,
                 **kwargs):
        # Set default file extension for images to .jpg if not specified
        kwargs.setdefault('file_extension', '.jpg')
        # Per-image normalization by default
        kwargs.setdefault('use_normalization', True)
        kwargs.setdefault('lower_percentile', 1.0)
        kwargs.setdefault('upper_percentile', 99.0)
        super().__init__(image_dir, mask_dir, **kwargs)
        
    def __getitem__(self, idx):
        """Override to handle the case where images are .jpg but masks are .png"""
        # Get the image filename
        image_file = self.image_files[idx]
        
        # For masks, try with different extensions
        # Extract the image file name without extension
        mask_file_base = os.path.splitext(image_file)[0]
        
        # Load image
        image_path = os.path.join(self.image_dir, image_file)
        
        # Try to find mask with .png, .jpg, or .jpeg extension
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate_path = os.path.join(self.mask_dir, mask_file_base + ext)
            if os.path.exists(candidate_path):
                mask_path = candidate_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {image_file} with .png/.jpg/.jpeg in {self.mask_dir}")
        
        # Load as PIL images
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load mask as grayscale
        
        # Store original RGB image for visualization (resize and normalize to [0, 1])
        original_image_rgb = image.resize((self.image_size, self.image_size))
        original_image_rgb = np.array(original_image_rgb).astype(np.float32) / 255.0
        
        # Convert to tensor (keep as RGB for visualization)
        original_tensor = torch.from_numpy(original_image_rgb).permute(2, 0, 1).float()
        
        # Apply transformations (returns PIL image + tensor mask)
        image_pil, mask_tensor = self.transform(image, mask)
        
        # Apply normalization on RAW pixel values (0-255)
        image_tensor = self.normalize(image_pil)
        
        # Return as dictionary to match base class format
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'original_image': original_tensor,
            'filename': image_file
        }


class ContrastEnhancedTransform:
    """
    Contrast enhancement pipeline for vessel segmentation.
    Applies CLAHE in LAB color space to preserve color while enhancing contrast.
    
    Pipeline:
    1. Convert RGB to LAB color space
    2. Apply CLAHE to L (Lightness) channel
    3. Apply gamma correction to L channel
    4. Convert back to RGB
    """
    def __init__(self, 
                 gamma: float = 1.2,
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8)):
        """
        Args:
            gamma: Gamma correction value (typically 1.1-1.3 for vessels)
            clahe_clip_limit: Clipping limit for CLAHE
            clahe_tile_size: Tile grid size for CLAHE
        """
        self.gamma = gamma
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, 
                                    tileGridSize=clahe_tile_size)
    
    def __call__(self, image):
        """Apply contrast enhancement pipeline to PIL Image."""
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle RGB images using LAB color space
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Split into L, A, B channels
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_channel = self.clahe.apply(l_channel)
            
            # Apply gamma correction to L channel
            l_channel = l_channel.astype(np.float32) / 255.0
            l_channel = np.power(l_channel, 1.0 / self.gamma)
            l_channel = (l_channel * 255).astype(np.uint8)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_channel, a_channel, b_channel])
            
            # Convert back to RGB
            image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(image_enhanced, mode='RGB')
        
        else:
            # Fallback for grayscale images
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply CLAHE
            image_8bit = (image * 255).astype(np.uint8)
            image_clahe = self.clahe.apply(image_8bit)
            image = image_clahe.astype(np.float32) / 255.0
            
            # Apply gamma correction
            image = np.power(image, 1.0 / self.gamma)
            
            # Convert back to PIL Image
            image_pil = (image * 255).astype(np.uint8)
            return Image.fromarray(image_pil, mode='L')


def create_dataloaders(dataset: Dataset, 
                      batch_size: int = 16, 
                      train_split: float = 0.8,
                      val_split: float = 0.1,
                      test_split: float = 0.1,
                      num_workers: int = 4,
                      random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders from a dataset.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training (default: 0.7)
        val_split: Fraction of data to use for validation (default: 0.15)
        test_split: Fraction of data to use for testing (default: 0.15)
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate splits sum to 1.0
    total_split = train_split + val_split + test_split
    if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Train/val/test splits must sum to 1.0, got {total_split}")
    
    # Calculate dataset sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size  # Remainder goes to test
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    
    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader