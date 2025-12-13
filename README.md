# CSE438 Semi-Supervised Learning Project

## Overview

This project implements **YOLOv8 instance segmentation for automated car damage detection** using multiple semi-supervised learning (SSL) techniques. The system combines deep learning with advanced data augmentation strategies to improve detection accuracy with limited labeled data.



## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended, CUDA 11.8+)
- Google Colab (for notebook execution) or local GPU machine




### 3. Dataset Setup

1. **Downloaded dataset** from Roboflow 
2. **Placed in data/ directory** with structure:
   ```
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```
3. **data.yaml format**:
   ```yaml
   path: (https://universe.roboflow.com/college-gxdrt/car-damage-detection-ha5mm/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
   train: (https://universe.roboflow.com/college-gxdrt/car-damage-detection-ha5mm/dataset/1/images?split=train)
   val: (https://universe.roboflow.com/college-gxdrt/car-damage-detection-ha5mm/dataset/1/images?split=valid)
   nc: 1  # number of classes
   names: ['car-damage']
   ```

## Full SSL Pipeline Explanation

### Stage 1: Baseline Model (assignment1_baseline/)

The baseline establishes a supervised learning foundation using YOLOv8 small instance segmentation model.

**Key Components:**
- **Model**: YOLOv8m-seg (medium instance segmentation)
- **Architecture**: Backbone → Neck → Head with segmentation branch
- **Training Strategy**: Standard supervised learning with labeled data

**Training Configuration:**
```python
model.train(
    data=DATA_YAML_PATH,
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20,
    device=0,
    amp=True
)
```

**Metrics Tracked:**
- Box loss (detection)
- Segmentation loss (mask prediction)
- Classification loss
- DFL loss (distribution focal loss)
- mAP50 & mAP50-95 (detection accuracy)

### Stage 2: Semi-Supervised Learning Methods (assignment2_ssl/)

#### FixMatch (fixmatch.ipynb)
**Principle**: Combines supervised loss on labeled data with consistency regularization on unlabeled data.

```python
# Pseudo-labeling with confidence threshold
if model_confidence > threshold:
    loss = supervised_loss + λ * consistency_loss
```

**Parameters:**
- Confidence threshold: 0.95
- Augmentation strength for weak/strong pairs
- Consistency weight (λ): 1.0

#### MixMatch (mixmatch1-1.ipynb)
**Principle**: Mixes inputs and labels from labeled and unlabeled data using manifold mixup.

```python
# For labeled data pair (x1, y1) and unlabeled pair (x2, pseudo_y2):
mixed_x = β·x1 + (1-β)·x2  # β ~ Beta(α, α)
mixed_y = β·y1 + (1-β)·pseudo_y2
loss = supervised(mixed_x, mixed_y) + consistency_loss
```

**Parameters:**
- Mix coefficient α: 0.75
- Consistency weight: 100
- Temperature (pseudo-labeling): 0.5

#### Mean Teacher (mean_teacher.ipynb)
**Principle**: Uses exponential moving average (EMA) of model weights as teacher for consistency regularization.

```python
# Student model: trained with labeled + unlabeled data
# Teacher model: θ_t = τ·θ_t + (1-τ)·θ_s (EMA update)
# Loss: supervised_loss + λ·MSE(student_pred, teacher_pred)
```

**Parameters:**
- EMA decay (τ): 0.999
- Consistency weight (λ): 10.0
- Ramp-up period: gradual increase of consistency weight

## How to Run Training and Evaluation

### Run Baseline Model
```bash
# Open assignment1_baseline/baseline.ipynb in Google Colab or Jupyter

# Mount Google Drive (in notebook)
from google.colab import drive
drive.mount('/content/drive')

# Training happens in notebook cells
# Run all cells in sequence
```

**Expected Output:**
- Training logs with loss curves
- Validation metrics every epoch
- Sample inference visualizations
- Best model saved to `results/weights/baseline_best.pt`

### Run Semi-Supervised Methods
```bash
# For each SSL method (FixMatch, MixMatch, Mean Teacher):

# 1. Open respective notebook in Google Colab
# 2. Set up data paths
# 3. Configure SSL parameters
# 4. Run training cells
# 5. Evaluate on validation set
```

**Expected Output:**
- SSL-specific training curves
- Comparison metrics vs baseline
- Pseudo-labeling quality analysis
- Final model weights

### Evaluate Models
```python
# In any notebook, after training:
from ultralytics import YOLO

model = YOLO('results/weights/best.pt')

# Run inference on validation set
results = model.predict(source='data/valid/images', conf=0.5)

# Get detailed metrics
metrics = results.box.all_metrics()
print(f"mAP50: {metrics['mAP50']:.3f}")
print(f"mAP50-95: {metrics['mAP50-95']:.3f}")
```

## Environment Setup

### Local Machine (GPU)
```bash
# Create virtual environment
python -m venv ssl_env
source ssl_env/bin/activate  # or ssl_env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Colab (Recommended)
```python
# Run in first notebook cell
!pip install -r /path/to/requirements.txt

# Mount Drive for data/results
from google.colab import drive
drive.mount('/content/drive')
```

## Key Parameters & Hyperparameters

### Model Configuration
| Parameter | Baseline | SSL Methods |
|-----------|----------|-------------|
| Model | YOLOv8s-seg | YOLOv8m-seg/YOLOv8m-seg |
| Input Size | 640×640 | 640×640 |
| Batch Size | 16 | 16 (8 labeled + 8 unlabeled) |
| Epochs | 50 | 50-100 |
| Optimizer | AdamW | AdamW |
| Learning Rate | 0.002 | 0.002 |

### Semi-Supervised Parameters
| Method | Key Parameter | Value | Purpose |
|--------|---------------|-------|---------|
| FixMatch | Confidence Threshold | 0.95 | Filter unreliable pseudo-labels |
| FixMatch | λ (consistency) | 1.0 | Balance labeled vs unlabeled loss |
| MixMatch | α (Beta distribution) | 0.75 | Control mixing intensity |
| MixMatch | Temperature | 0.5 | Sharpen pseudo-label distribution |
| Mean Teacher | τ (EMA decay) | 0.999 | Teacher update smoothness |
| Mean Teacher | λ (consistency) | 10.0 | Consistency regularization weight |


## Troubleshooting

### Out of Memory (OOM) Error
```python
# Reduce batch size in notebook
imgsz = 512  # or 384
batch_size = 8  # or 4
```

### Dataset Not Found
```python
# Verify path in data.yaml
# Ensure images are in correct subdirectories
# Check file naming consistency
```

### Poor SSL Performance
1. **Check pseudo-label quality**: Lower confidence threshold → more pseudo-labels
2. **Adjust λ (consistency weight)**: Too high → overfitting on weak labels
3. **Verify augmentation strength**: Too weak → no benefit from unlabeled data
4. **Check labeled/unlabeled ratio**: Usually 1:3 to 1:10 works best

## Project Roadmap

1. ✅ **Assignment 1**: Implement supervised baseline with YOLOv8
2. ✅ **Assignment 2**: Implement FixMatch, MixMatch, Mean Teacher
3. ⏳ **Final Project**: 
   - Compare all SSL methods
   - Optimize best-performing technique
   - Create final submission with visualizations

## References

### YOLOv8 & Instance Segmentation
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- YOLOv8: A State-of-the-Art Object Detector





**Python Version**: 3.8+  
**Framework**: PyTorch 2.0+, Ultralytics 8.3+
