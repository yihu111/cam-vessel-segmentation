"""
Evaluate trained vessel segmentation models and save prediction visualizations.
"""

import argparse
import torch
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.lightning.models import SegmentationModel
from src.utils.evaluation import evaluate_model
from src.data.datasets import FundusDataset, CAMDataset
from torch.utils.data import DataLoader, Subset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation model and save visualizations.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with images/ and masks/ subfolders')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results_eval', help='Directory to save visualizations')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to visualize/save')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'cam'], help='Dataset type')
    parser.add_argument('--eval_set', type=str, default='test', choices=['test', 'val', 'full'], 
                        help='Which dataset to evaluate: test (default), val (validation only), or full (all data)')
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SegmentationModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    # Prepare dataset and loader
    if args.dataset == 'fundus':
        dataset_class = FundusDataset
    else:
        dataset_class = CAMDataset

    # Load full dataset
    full_dataset = dataset_class(
        image_dir=f"{args.data_dir}/images",
        mask_dir=f"{args.data_dir}/masks",
        image_size=args.image_size,
        augmentation=False,
        use_normalization=True
    )
    
    # Determine which dataset to evaluate based on mode
    checkpoint_dir = os.path.dirname(args.checkpoint)
    split_file = os.path.join(checkpoint_dir, f'split_indices_{args.dataset}.json')
    
    if args.eval_set == 'full':
        # Evaluate on full dataset
        print("Evaluating on full dataset")
        eval_dataset = full_dataset
        set_name = "Full Dataset"
        
    elif args.eval_set == 'val':
        # Evaluate on validation set only
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            val_indices = split_info['val_indices']
            eval_dataset = Subset(full_dataset, val_indices)
            print(f"Evaluating on validation set: {len(eval_dataset)} images")
            set_name = "Validation Set"
        else:
            print(f"Warning: No split file found at {split_file}")
            print("Cannot use validation mode without split file. Using full dataset.")
            eval_dataset = full_dataset
            set_name = "Full Dataset"
            
    else:  # test mode (default)
        # Evaluate on test set only
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            test_indices = split_info['test_indices']
            eval_dataset = Subset(full_dataset, test_indices)
            print(f"Evaluating on test set: {len(eval_dataset)} images")
            set_name = "Test Set"
        else:
            print(f"Warning: No split file found at {split_file}")
            print("Cannot use test mode without split file. Using full dataset.")
            eval_dataset = full_dataset
            set_name = "Full Dataset"
    
    # Create data loader
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate and save visualizations
    metrics = evaluate_model(
        model=model,
        test_loader=eval_loader,
        device=device,
        threshold=args.threshold,
        save_dir=args.output_dir,
        save_predictions=True,
        num_samples_to_save=args.num_images,
        set_name=set_name
    )
    
    print(f"Evaluated on {len(eval_dataset)} images")
    print(f"{set_name} metrics:", metrics)

if __name__ == "__main__":
    main()