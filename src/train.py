"""
Train vessel segmentation model using PyTorch Lightning with YAML configuration.
"""


import os
import sys
import argparse
from pathlib import Path
import yaml
import json
import shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import custom modules
from src.lightning.models import SegmentationModel
from src.data.datasets import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train vessel segmentation model with PyTorch Lightning')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for model checkpoints')
    
    # Allow overriding any config option via command line
    parser.add_argument('--override', nargs='*', default=[], 
                      help='Override config values, e.g. --override model.encoder=resnet50 training.batch_size=16')
    
    return parser.parse_args()

def load_config(config_path):
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def override_config(config, overrides):
    """Override config values from command line arguments"""
    for override in overrides:
        if '=' not in override:
            continue
        key, value = override.split('=', 1)
        keys = key.split('.')
        
        # Navigate to the nested dictionary
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Convert value to the right type
        try:
            # Try to convert to numeric if possible
            if value.lower() == 'true':
                typed_value = True
            elif value.lower() == 'false':
                typed_value = False
            elif value.lower() == 'null' or value.lower() == 'none':
                typed_value = None
            else:
                try:
                    typed_value = int(value)
                except ValueError:
                    try:
                        typed_value = float(value)
                    except ValueError:
                        typed_value = value
            
            # Set the value
            current[keys[-1]] = typed_value
            print(f"Overriding {key} with {typed_value}")
            
        except Exception as e:
            print(f"Error overriding {key}: {e}")
    
    return config

def setup_training(config, output_dir=None):
    """Setup training components from config"""
    
    # Set up output directory
    if output_dir is None:
        encoder = config['model'].get('encoder', config['model'].get('foundation_model', 'none'))
        output_dir = os.path.join(
            config.get('logging', {}).get('log_dir', 'lightning_logs'),
            f"{config['model']['name']}_{encoder}"
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the config for reproducibility
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set up model
    print(f"Model: {config['model']['name']} with {config['model']['encoder']}")
    model = SegmentationModel(
        model_name=config['model']['name'],
        encoder_name=config['model']['encoder'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        threshold=float(config['training']['threshold']),
        dice_weight=float(config['training']['loss']['dice_weight']),
        focal_weight=float(config['training']['loss']['focal_weight']),
        pretrained_weights=config['model'].get('pretrained_weights'),
        freeze_encoder=config['model'].get('freeze_encoder', False)
    )
    
    # Set up dataset
    if 'cam' in config['data']['data_dir'].lower():
        from src.data.datasets import CAMDataset
        dataset = CAMDataset(
            image_dir=os.path.join(config['data']['data_dir'], 'images'),
            mask_dir=os.path.join(config['data']['data_dir'], 'masks'),
            image_size=config['data']['image_size'],
            augmentation=config['data'].get('augmentation', True),
            gamma=config['data']['contrast_enhancement'].get('gamma', 1.2),
            clahe_clip_limit=config['data']['contrast_enhancement'].get('clahe_clip_limit', 2.0),
            clahe_tile_size=tuple(config['data']['contrast_enhancement'].get('clahe_tile_size', (8, 8))),
            file_extension='.jpg',
            use_normalization=config['data'].get('use_normalization', True),
            lower_percentile=config['data'].get('lower_percentile', 1.0),
            upper_percentile=config['data'].get('upper_percentile', 99.0)
        )
        print(f"Using CAMDataset for {config['data']['data_dir']}")
    else:
        from src.data.datasets import FundusDataset
        dataset = FundusDataset(
            image_dir=os.path.join(config['data']['data_dir'], 'images'),
            mask_dir=os.path.join(config['data']['data_dir'], 'masks'),
            image_size=config['data']['image_size'],
            augmentation=config['data'].get('augmentation', True),
            gamma=config['data']['contrast_enhancement'].get('gamma', 1.2),
            clahe_clip_limit=config['data']['contrast_enhancement'].get('clahe_clip_limit', 2.0),
            clahe_tile_size=tuple(config['data']['contrast_enhancement'].get('clahe_tile_size', (8, 8))),
            use_normalization=config['data'].get('use_normalization', True),
            lower_percentile=config['data'].get('lower_percentile', 1.0),
            upper_percentile=config['data'].get('upper_percentile', 99.0)
        )
        print(f"Using FundusDataset for {config['data']['data_dir']}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['splits']['train'],
        val_split=config['data']['splits']['val'],
        test_split=config['data']['splits']['test'],
        num_workers=config['hardware']['num_workers']
    )
    
    # Determine dataset name for split indices
    if 'cam' in config['data']['data_dir'].lower():
        dataset_name = 'cam'
    else:
        dataset_name = 'fundus'
    
    # Save split indices for later evaluation
    split_info = {
        'train_indices': train_loader.dataset.indices,
        'val_indices': val_loader.dataset.indices,
        'test_indices': test_loader.dataset.indices,
        'dataset': dataset_name,
        'total_size': len(dataset)
    }
    
    split_file = os.path.join(output_dir, f'split_indices_{dataset_name}.json')
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split indices to {split_file}")
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{{epoch}}-{{val_dice:.4f}}",
        monitor="val_dice",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=config['training']['patience'],
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    # Set up logger
    encoder = config['model'].get('encoder', config['model'].get('foundation_model', 'none'))
    logger = TensorBoardLogger(
        save_dir=config['logging']['log_dir'],
        name=f"{config['model']['name']}_{encoder}",
        version=None,
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        accelerator="gpu" if config['hardware']['gpus'] > 0 else "cpu",
        devices=config['hardware']['gpus'] if config['hardware']['gpus'] > 0 else None,
        precision=config['hardware']['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=True,
        gradient_clip_val=config['training'].get('gradient_clip_val', 0.0),
    )
    
    return trainer, model, train_loader, val_loader, test_loader, checkpoint_callback

def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    print("Seed set to 42")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    config = override_config(config, args.override)
    
    # Setup training components
    trainer, model, train_loader, val_loader, test_loader, checkpoint_callback = setup_training(config, args.output_dir)
    
    # Print training info
    encoder = config['model'].get('encoder', config['model'].get('foundation_model', 'none'))
    print(f"\nTraining {config['model']['name']} with {encoder} encoder")
    print(f"Data directory: {config['data']['data_dir']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Device: {'GPU' if config['hardware']['gpus'] > 0 else 'CPU'}")
    print(f"Precision: {config['hardware']['precision']}\n")
    
    # Print dataset splits
    print(f"Data splits:")
    print(f"  Train: {len(train_loader.dataset)} images")
    print(f"  Validation: {len(val_loader.dataset)} images")
    print(f"  Test: {len(test_loader.dataset)} images\n")
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    test_results = trainer.test(model, test_loader)
    
    # Create best.ckpt as a copy of the best model
    best_path = checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        output_dir = os.path.dirname(best_path)
        best_link = os.path.join(output_dir, 'best.ckpt')
        shutil.copy(best_path, best_link)
        print(f"\nBest model copied to: {best_link}")
    
    print("\nTraining completed!")
    print(f"Test results: {test_results}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation score: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    main()
