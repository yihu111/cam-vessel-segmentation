# Vessel Segmentation for CAM Images

This repository contains the complete pipeline for training and evaluating vessel segmentation models on Chorioallantoic Membrane (CAM) images using transfer learning from fundus vessel segmentation. The project uses PyTorch Lightning, Segmentation Models PyTorch (SMP), and supports multiple architectures including UNet, UNet++, and DeepLabV3+ with various encoder backbones.

---

## Overview

This project implements a two-stage transfer learning approach:

1. **Stage 1**: Train models on retinal fundus vessel segmentation datasets
2. **Stage 2**: Fine-tune the trained models on CAM (Chorioallantoic Membrane) vessel images

The pipeline supports:
- Multiple segmentation architectures (UNet, UNet++, DeepLabV3+)
- Various encoder backbones (ResNet, EfficientNet, MiT transformers)
- Advanced data augmentation and preprocessing
- PyTorch Lightning for efficient training
- Comprehensive evaluation with visualizations
- ONNX export for deployment

---

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cam-vessel-segmentation
```

### Step 2: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch 2.6.0+ with CUDA support
- PyTorch Lightning 2.5.2+
- Segmentation Models PyTorch 0.5.0
- Albumentations 2.0.8 (data augmentation)
- ONNX and ONNXRuntime (for model export)

---

## Data Preparation

### Download Datasets

**Note:** The repository includes empty `data/processed/` folders. You need to download the actual data separately.

#### 1. Fundus Vessel Segmentation Dataset

Download the fundus dataset from:
- **FIVES Dataset**: [Nature Scientific Data - A large-scale optical microscope retinal vessel image dataset for intelligent fundus vessel segmentation](https://www.nature.com/articles/s41597-022-01564-3)

Place the images and masks in:
```
data/processed/fundus/
├── images/
└── masks/
```

#### 2. CAM Vessel Segmentation Dataset

**Download the CAM dataset** (fine-tuning data and optional evaluation datasets) from:

**[https://drive.google.com/drive/folders/1QAHu881phWsjoaVviETrAKwnajDh85YW?usp=sharing](https://drive.google.com/drive/folders/1QAHu881phWsjoaVviETrAKwnajDh85YW?usp=sharing)**

The download includes:
- **fine-tuning/** - Training data for CAM vessel segmentation (required)
- **ED11/** - Evaluation dataset for Day 11 embryos (optional, for quantification analysis)
- **ED15/** - Evaluation dataset for Day 15 embryos (optional, for quantification analysis)

**For training**, only the **fine-tuning** folder is required. Place it as follows:
```
data/processed/cam/
├── images/          # From fine-tuning/images
└── masks/           # From fine-tuning/masks
```

ED11 and ED15 datasets can be used for exploring quantification results on segmentation performance of the final model across different embryonic stages.

**Note:** Images 01 to 05 in both ED11 and ED15 datasets are training images and should not be used for biological validation purposes.

---

## Pre-built Applications

**For quick automated segmentation and quantification without manual setup**, download the ready-to-use applications:

**[Download Pre-built Apps (AppImage for Linux & .exe for Windows)](https://drive.google.com/drive/folders/1wiYDJk_f6glJ8CU_X81DkwIPEdlIxG0F?usp=drive_link)**

- **Linux**: AppImage - No installation required, just make executable and run
- **Windows**: .exe installer - Double-click to install and launch

These applications provide automated vessel segmentation and quantification with results generated automatically.

---

## Usage

### 1. Training on Fundus Dataset

**First, train a model on the fundus vessel segmentation dataset.** This serves as the pre-trained model for transfer learning to CAM images.

#### Example: Train UNet with MIT-B1 encoder

```bash
python src/train.py --config configs/fundus.yaml
```

This will:
- Load configuration from [configs/fundus.yaml](configs/fundus.yaml)
- Create dataloaders with train/val/test splits (80%/10%/10%)
- Train the model with early stopping (patience=8)
- Save checkpoints to `lightning_logs_fundus_mit_b1/unet_mit_b1/`
- Save best model as `best.ckpt`

#### Training Output

Training produces:
- `best.ckpt` - Best model based on validation Dice score
- `last.ckpt` - Last epoch checkpoint
- `config.yaml` - Copy of training configuration
- `split_indices_fundus.json` - Train/val/test split indices
- `version_0/` - TensorBoard logs and training curves

---

### 2. Transfer Learning to CAM Dataset

**After training on fundus, fine-tune the model on CAM images.**

#### Example: Fine-tune MIT-B1 model on CAM data

```bash
python src/train.py --config configs/cam.yaml
```

#### Training Output

Similar to fundus training, but saves to:
- `lightning_logs_cam_mit_b1/unet_mit_b1/`
- Uses `split_indices_cam.json` for reproducible splits

---

### 3. Model Evaluation

**Evaluate trained models and generate visualizations.**

#### Evaluate on Test Set

```bash
python src/eval.py \
  --checkpoint lightning_logs_cam_mit_b1/unet_mit_b1/best.ckpt \
  --data_dir data/processed/cam \
  --dataset cam \
  --eval_set test \
  --output_dir results_cam_eval \
  --num_images 20 \
  --image_size 1024 \
  --threshold 0.5
```

#### Evaluation Arguments

- `--checkpoint`: Path to trained model checkpoint (`.ckpt` file)
- `--data_dir`: Directory containing `images/` and `masks/` subfolders
- `--dataset`: Dataset type (`fundus` or `cam`)
- `--eval_set`: Evaluation mode:
  - `test` - Evaluate on test set only (default)
  - `val` - Evaluate on validation set only
  - `full` - Evaluate on entire dataset
- `--output_dir`: Directory to save visualizations and results
- `--num_images`: Number of images to save visualizations for
- `--image_size`: Image size for evaluation (should match training)
- `--threshold`: Binary segmentation threshold (default: 0.5)

---

### 4. ONNX Export

**Export trained PyTorch models to ONNX format for deployment.**

#### Export with Dynamic Input Size

```bash
python scripts/to_onnx.py \
  --checkpoint lightning_logs_cam_mit_b1/unet_mit_b1/best.ckpt \
  --output vessel_model_cam_mit_b1.onnx \
  --input_size 1024 \
  --dynamic_axes \
  --simplify
```

#### ONNX Export Arguments

- `--checkpoint`: Path to PyTorch Lightning checkpoint
- `--output`: Output ONNX file path
- `--input_size`: Reference input size (used for tracing, default: 1024)
- `--dynamic_axes`: Enable dynamic batch size and spatial dimensions
- `--opset_version`: ONNX opset version (default: 14)
- `--simplify`: Simplify ONNX graph (requires `onnx-simplifier`)

#### Using Dynamic Axes

With `--dynamic_axes`, the exported model supports:
- Variable batch sizes
- Variable input heights and widths
- Useful for inference on images of different sizes

#### Verify ONNX Model

```python
import onnx
model = onnx.load("vessel_model_cam_mit_b1.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid!")
```

#### Run ONNX Inference

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("vessel_model_cam_mit_b1.onnx")

# Prepare input (B, C, H, W)
input_data = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_data})
prediction = outputs[0]  # Shape: (1, 1, 1024, 1024)
```

---

## Configuration Files

### YAML Configuration Structure

Configuration files control all aspects of training and model architecture.

#### [configs/fundus.yaml](configs/fundus.yaml)

```yaml
# Dataset settings
data:
  data_dir: "data/processed/fundus"
  image_size: 1024
  augmentation: true         # Enable data augmentation
  use_normalization: true    # Percentile-based normalization
  lower_percentile: 1.0
  upper_percentile: 99.0
  contrast_enhancement:
    gamma: 1.2
    clahe_clip_limit: 2.0
    clahe_tile_size: [8, 8]
  splits:
    train: 0.8
    val: 0.1
    test: 0.1

# Model settings
model:
  name: "unet"               # unet, unetplusplus, deeplabv3plus
  encoder: "mit_b1"          # resnet34, efficientnet-b0, mit_b1, mit_b2
  in_channels: 3             # RGB images
  classes: 1                 # Binary segmentation
  pretrained_weights: null   # No pretrained weights for fundus

# Training settings
training:
  batch_size: 2
  epochs: 30
  learning_rate: 1e-4
  weight_decay: 1e-5
  gradient_clip_val: 0.5
  loss:
    dice_weight: 0.5         # Dice loss weight
    focal_weight: 0.5        # Focal loss weight
  threshold: 0.5
  patience: 8                # Early stopping patience
  seed: 42

# Hardware settings
hardware:
  gpus: 1
  num_workers: 4
  precision: 32

# Logging settings
logging:
  log_dir: "lightning_logs_fundus_mit_b1"
```

#### [configs/cam.yaml](configs/cam.yaml)

Similar structure to `fundus.yaml`, but with key differences:

```yaml
model:
  pretrained_weights: "lightning_logs_fundus_mit_b1/unet_mit_b1/best.ckpt"
  freeze_encoder: false

training:
  learning_rate: 1e-5        # Lower learning rate for transfer learning
```

### Overriding Configuration via Command Line

You can override any configuration parameter without modifying the YAML file:

```bash
python src/train.py --config configs/fundus.yaml \
  --override model.encoder=resnet34 \
             training.batch_size=4 \
             training.learning_rate=5e-5 \
             logging.log_dir=lightning_logs_custom
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
