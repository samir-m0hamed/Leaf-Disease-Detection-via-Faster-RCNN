# Leaf Disease Detection using Faster R-CNN

<div dir="rtl">

**كشف أمراض الأوراق باستخدام Faster R-CNN - نموذج متقدم لكشف الأجسام من عائلة R-CNN**

</div>

A complete deep learning implementation for detecting and localizing leaf diseases using **Faster R-CNN with ResNet50 + FPN backbone**. This project demonstrates the end-to-end pipeline for object detection in agricultural applications.

## 🎯 Project Overview

This repository implements a state-of-the-art object detection system specifically designed for identifying diseased leaves in images. The model uses:

- **Architecture**: Faster R-CNN with ResNet50 backbone + Feature Pyramid Networks (FPN)
- **Framework**: PyTorch with TorchVision
- **Dataset Format**: COCO (Common Objects in Context)
- **Device Support**: GPU (CUDA) or CPU

## 📁 Project Structure

```
.
├── Dataset/                                    # COCO format leaf disease dataset
│   ├── train/                                 # Training images & annotations
│   │   ├── _annotations.coco.json
│   │   └── [image files]
│   ├── valid/                                 # Validation split
│   │   ├── _annotations.coco.json
│   │   └── [image files]
│   ├── test/                                  # Test split
│   │   ├── _annotations.coco.json
│   │   └── [image files]
│   ├── README.dataset.txt                    # Dataset metadata
│   └── README.roboflow.txt                   # Roboflow export info
├── Leaf Disease Detection via faster R-CNN/  # Main notebook with full pipeline
│   └── Leaf_Disease_Detection.ipynb
└── README.md                                  # This file
```

## 🔄 Complete Pipeline

### **Stage 1: Data Preparation**
- Load COCO format annotations from JSON files
- Parse category information (disease types)
- Extract bounding boxes and class labels
- Handle image-to-annotation mapping

### **Stage 2: Custom Dataset Class**
- **COCODataset class**: Loads images and converts COCO format `[x, y, w, h]` → `[x1, y1, x2, y2]`
- Handles images with multiple disease objects per image
- Returns image tensors and target dictionaries with boxes and labels
- Supports variable number of objects per image

### **Stage 3: Data Loading**
- Custom collate function for handling variable-size batches
- DataLoader with configurable batch sizes
- Support for both GPU and CPU training
- Multi-worker loading for efficiency

### **Stage 4: Model Configuration**
- Pre-trained Faster R-CNN with ResNet50 backbone
- Feature Pyramid Network (FPN) for multi-scale detection
- Transfer learning optimization
- Fine-tuning on leaf disease dataset

### **Stage 5: Training Loop**
- Forward pass: Image → RPN → ROI pooling → Classification & Regression
- Loss calculation: Classification loss + Bounding box regression loss
- Optimizer: Typically Adam or SGD
- Learning rate scheduling and warmup strategies

### **Stage 6: Validation & Evaluation**
- Inference on validation set
- Computation of detection metrics (AP, mAP)
- Bounding box visualization
- Confidence score filtering

### **Stage 7: Testing & Inference**
- Inference on unseen test images
- Real-time disease detection
- Visualization with predicted boxes and confidence scores

### **Stage 8: Post-Processing**
- Non-Maximum Suppression (NMS) for overlapping boxes
- Confidence thresholding
- Disease classification from detected objects

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional, CPU works too)
- TorchVision
- Required packages: OpenCV, PIL, NumPy, Matplotlib

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/leaf-disease-detection.git
cd leaf-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install opencv-python pillow numpy matplotlib tqdm
```

### Running the Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open: Leaf Disease Detection via faster R-CNN/Leaf_Disease_Detection.ipynb
# Run cells sequentially for full pipeline
```

## 📊 Configuration Parameters

```python
# In the notebook (Step 2)
BATCH_SIZE = 4              # Smaller for local GPU, larger for cloud
NUM_WORKERS = 0             # Set to 0 on Windows, >0 on Linux
NUM_EPOCHS = 10             # Recommend 10-50 for better results
LEARNING_RATE = 0.001       # 0.001 for fine-tuning, adjust as needed
DEVICE = 'cuda' / 'cpu'     # Auto-detected based on availability
```

## 🎓 Key Concepts

### Faster R-CNN Architecture
1. **Region Proposal Network (RPN)**: Generates candidate bounding boxes
2. **ROI Pooling**: Extracts fixed-size features for each proposal
3. **Classification Head**: Predicts disease type for each ROI
4. **Regression Head**: Refines bounding box coordinates

### COCO Dataset Format
- Stores images and annotations separately
- Bounding boxes in `[x, y, width, height]` format
- Supports multiple annotations per image
- JSON files for easy parsing and version control

### Transfer Learning
- Pre-trained ResNet50 backbone (ImageNet weights)
- Feature Pyramid Network for multi-scale features
- Faster convergence with less data
- Better generalization to new diseases

## 📈 Expected Results

After proper training:
- **Validation mAP@0.5**: 0.70-0.85
- **Validation mAP@[0.5:0.95]**: 0.50-0.70
- **Inference time**: ~100-200ms per image (GPU)
- **Model size**: ~170 MB (ResNet50+FPN)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `BATCH_SIZE` or use gradient accumulation |
| Slow training | Enable GPU (CUDA), increase `NUM_WORKERS` |
| Poor accuracy | More epochs, data augmentation, learning rate scheduling |
| COCO parsing error | Verify annotation JSON structure with `python -m json.tool` |

## 📚 Dataset Information

The dataset uses the COCO format with:
- **Train split**: Main training data
- **Validation split**: Model selection during training
- **Test split**: Final evaluation on unseen data
- **Categories**: Multiple leaf disease types (see `categories_dict` output)

For dataset details, see:
- `Dataset/README.dataset.txt` - Dataset metadata
- `Dataset/README.roboflow.txt` - Export information from Roboflow

## 🛠 Technologies Used

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities and pre-trained models
- **OpenCV**: Image processing
- **NumPy/Matplotlib**: Data manipulation and visualization
- **Jupyter**: Interactive notebook environment

## 📖 References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [COCO Dataset Format](https://cocodataset.org/)
- [PyTorch Detection](https://pytorch.org/vision/stable/models.html#detection)

## 👤 Author

AI Computer Vision Projects - Object Detection Series

## 📝 License

This project is open source and available for educational purposes.

---

<div dir="rtl">

### ملاحظات بالعربية 🇪🇬

هذا المشروع يغطي **pipeline كامل** لكشف الأمراض في الأوراق:
- تحضير البيانات بصيغة COCO
- تصميم custom dataset class
- بناء نموذج Faster R-CNN
- تدريب وتقييم النموذج
- الاستدلال على صور جديدة

المشروع مناسب للتعلم العملي وفهم كيفية تطبيق object detection في الزراعة الذكية.

</div>

