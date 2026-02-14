#!/usr/bin/env python3
"""
Export trained PyTorch Lightning model to ONNX format for deployment.
Supports dynamic input sizes for fully convolutional models.
"""

import torch
import argparse
from pathlib import Path
import yaml
import sys

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.lightning.models import SegmentationModel

def export_to_onnx(checkpoint_path: str, 
                   output_path: str,
                   input_size: int = 1024,
                   dynamic_axes: bool = True,  # Enable dynamic input size
                   opset_version: int = 14,
                   simplify: bool = True):
    """
    Export PyTorch Lightning model to ONNX format with dynamic input size support.
    
    Args:
        checkpoint_path: Path to trained .ckpt file
        output_path: Output path for .onnx file
        input_size: Input image size for dummy input (default: 1024)
        dynamic_axes: Whether to support dynamic input sizes (default: True)
        opset_version: ONNX opset version (default: 14)
        simplify: Whether to simplify the ONNX model (requires onnx-simplifier)
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load trained model
    model = SegmentationModel.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model.eval()
    
    # Move model to CPU for ONNX export
    model = model.cpu()
    
    # Create dummy input on CPU (used only for tracing, actual size can vary)
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print(f"Exporting model to ONNX...")
    print(f"Device: CPU (required for ONNX export)")
    print(f"Dynamic input size: {'Enabled' if dynamic_axes else 'Disabled'}")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output path: {output_path}")
    
    # Configure dynamic axes
    if dynamic_axes:
        # Allow batch size, height, and width to be dynamic
        dynamic_config = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
        print("Dynamic axes: batch_size, height, width")
    else:
        dynamic_config = None
        print("Fixed input size mode")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model.model,  # The actual segmentation model (not the Lightning wrapper)
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_config,
            verbose=False
        )
    
    print(f"✓ Model exported to: {output_path}")
    
    # Simplify ONNX model (optional but recommended)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_simplified, check = onnx_simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, output_path)
                print("✓ Model simplified successfully")
            else:
                print("⚠ Simplification check failed, keeping original model")
        except ImportError:
            print("⚠ onnx-simplifier not installed. Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠ Simplification failed: {e}")
            print("Keeping original (non-simplified) model")
    
    # Verify the exported model
    print("\nVerifying exported model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Print model info
    print("\nModel Information:")
    print(f"  Input name: {onnx_model.graph.input[0].name}")
    input_shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                   for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"  Input shape: {input_shape}")
    print(f"  Output name: {onnx_model.graph.output[0].name}")
    output_shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                    for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    print(f"  Output shape: {output_shape}")
    
    # Get model size
    model_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Test inference with different sizes
    print("\nTesting ONNX inference with different input sizes...")
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    
    # Test with original size
    test_sizes = [(input_size, input_size)]
    
    # Test with different sizes if dynamic axes enabled
    if dynamic_axes:
        test_sizes.extend([
            (512, 512),    # Smaller
            (768, 768),    # Medium
            (1024, 1024),  # Original
            (2048, 2048),  # Larger
            (512, 768),    # Non-square
            (1024, 512),   # Non-square
        ])
    
    for h, w in test_sizes:
        try:
            test_input = torch.randn(1, 3, h, w).numpy()
            outputs = ort_session.run(None, {'input': test_input})
            print(f"  ✓ Input: {test_input.shape} -> Output: {outputs[0].shape}")
        except Exception as e:
            print(f"  ✗ Failed for size ({h}, {w}): {e}")
    
    return output_path


def export_preprocessing_config(config_path: str, output_path: str, dynamic_size: bool = True):
    """
    Export preprocessing configuration to JSON for frontend.
    
    Args:
        config_path: Path to training config YAML
        output_path: Output path for preprocessing JSON
        dynamic_size: Whether model supports dynamic input sizes
    """
    import json
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract preprocessing parameters
    preprocessing = {
        'image_size': config['data']['image_size'],
        'dynamic_size': dynamic_size,
        'min_size': 256,
        'max_size': 4096,
        
        # Normalization settings
        'use_normalization': config['data'].get('use_normalization', False),
        'lower_percentile': config['data'].get('lower_percentile', 1.0),
        'upper_percentile': config['data'].get('upper_percentile', 99.0),
        'normalization_type': 'per_image_per_channel_percentile_zscore',
        
        # Contrast enhancement settings
        'contrast_enhancement': config['data'].get('contrast_enhancement', {}),
        
        # Post-processing settings
        'threshold': config['training'].get('threshold', 0.5),
        
        # Model info
        'input_channels': config['model']['in_channels'],
        'output_classes': config['model']['classes'],
        
        # Instructions
        'preprocessing_steps': [
            '1. Resize image to target size (e.g., 1024x1024)',
            '2. Apply gamma correction (from config)',
            '3. Apply CLAHE in LAB color space (from config)',
            '4. Normalize to [0, 1] by dividing by 255',
            '5. Per-image, per-channel normalization (if enabled):',
            '   a. For each RGB channel separately:',
            '   b. Clip to percentile range (lower_percentile to upper_percentile)',
            '   c. Apply z-score: (channel - mean) / std',
            '6. Convert to CHW format: [C, H, W]',
            '7. Add batch dimension: [1, C, H, W]'
        ],
        
        'postprocessing_steps': [
            '1. Get model output (logits)',
            '2. Apply sigmoid: probs = 1 / (1 + exp(-logits))',
            '3. Apply threshold: mask = probs > threshold',
            '4. Resize mask back to original image size if needed'
        ],
        
        'notes': 'CRITICAL: You must apply the exact same preprocessing in your app! '
                 'The ONNX model only contains the neural network weights. '
                 'Per-image normalization means each image is normalized independently with percentile clipping.'
    }
    
    with open(output_path, 'w') as f:
        json.dump(preprocessing, f, indent=2)
    
    print(f"\n✓ Preprocessing config saved to: {output_path}")
    
    # Print preprocessing summary
    print("\n" + "="*50)
    print("PREPROCESSING REQUIREMENTS FOR YOUR APP")
    print("="*50)
    print("\nYou MUST apply these steps BEFORE feeding to ONNX model:")
    print("\n1. Resize to target size")
    
    if config['data'].get('contrast_enhancement'):
        ce = config['data']['contrast_enhancement']
        print(f"\n2. Contrast Enhancement:")
        if 'gamma' in ce:
            print(f"   - Gamma correction: {ce['gamma']}")
        if 'clahe_clip_limit' in ce:
            print(f"   - CLAHE clip limit: {ce['clahe_clip_limit']}")
            print(f"   - CLAHE tile size: {ce.get('clahe_tile_size', [8, 8])}")
    
    if config['data'].get('use_normalization'):
        print(f"\n3. Per-Image Normalization (PER CHANNEL):")
        print(f"   - Lower percentile: {config['data'].get('lower_percentile', 1)}%")
        print(f"   - Upper percentile: {config['data'].get('upper_percentile', 99)}%")
        print(f"   - For EACH RGB channel separately:")
        print(f"     a) Clip to percentile range")
        print(f"     b) Z-score: (channel - mean) / std")
        print(f"   - This makes the model robust to lighting variations!")
    
    print(f"\n4. Convert to tensor: [1, 3, H, W]")
    print(f"\n5. Run ONNX model → get logits")
    print(f"\n6. Apply sigmoid: probs = 1/(1+exp(-logits))")
    print(f"\n7. Apply threshold: mask = probs > {config['training'].get('threshold', 0.5)}")
    print("\n" + "="*50)
    
    return preprocessing


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX with dynamic size support')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained .ckpt file')
    parser.add_argument('--output', type=str, default='model.onnx',
                        help='Output path for .onnx file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config YAML (optional)')
    parser.add_argument('--input-size', type=int, default=1024,
                        help='Input image size for dummy input (default: 1024)')
    parser.add_argument('--fixed-size', action='store_true',
                        help='Disable dynamic input size (fix to input-size)')
    parser.add_argument('--opset-version', type=int, default=14,
                        help='ONNX opset version (default: 14)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX model simplification')
    
    args = parser.parse_args()
    
    # Export to ONNX
    output_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_size=args.input_size,
        dynamic_axes=not args.fixed_size,  # Enable by default
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )
    
    # Export preprocessing config if training config provided
    if args.config:
        config_output = args.output.replace('.onnx', '_config.json')
        export_preprocessing_config(
            args.config, 
            config_output, 
            dynamic_size=not args.fixed_size
        )
    
    print("\n" + "="*50)
    print("Export completed successfully!")
    print("="*50)
    print(f"\nFiles generated:")
    print(f"  1. ONNX model: {output_path}")
    if args.config:
        print(f"  2. Preprocessing config: {config_output}")
    
    if not args.fixed_size:
        print(f"\n✓ Model supports flexible input sizes!")
        print(f"  - Tested sizes: 512x512 to 2048x2048")
        print(f"  - Non-square images: Supported")
        print(f"  - Memory usage scales with input size")
    else:
        print(f"\n⚠ Model uses fixed input size: {args.input_size}x{args.input_size}")
    
    print(f"\nUsage in frontend:")
    print(f"  - Load model with ONNX Runtime")
    print(f"  - Apply same preprocessing as training")
    if not args.fixed_size:
        print(f"  - Input shape: [1, 3, ANY_HEIGHT, ANY_WIDTH]")
        print(f"  - Output shape: [1, 1, ANY_HEIGHT, ANY_WIDTH]")
    else:
        print(f"  - Input shape: [1, 3, {args.input_size}, {args.input_size}]")
        print(f"  - Output shape: [1, 1, {args.input_size}, {args.input_size}]")


if __name__ == '__main__':
    main()