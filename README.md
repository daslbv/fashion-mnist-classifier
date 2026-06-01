# Fashion-MNIST Classifier

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

Classify fashion items into 10 categories (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot) using a CNN built with PyTorch.

> **Problem:** Can we automatically identify clothing items from a grayscale image? This project demonstrates a complete computer vision pipeline — from CNN architecture design to training to a web app for real-time prediction.

---

## Dataset

**Source:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — Zalando's fashion image dataset

| Stat | Value |
|---|---|
| Training samples | 1,000 (100 per class) |
| Test samples | 10,000 |
| Image size | 28x28 grayscale |
| Classes | 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |

---

## Model Architecture

```
Input (1x28x28 grayscale)
  -> Conv2D 32 + BatchNorm + ReLU + MaxPool2x2
  -> Conv2D 64 + BatchNorm + ReLU + MaxPool2x2
  -> Conv2D 128 + BatchNorm + ReLU + MaxPool2x2
  -> Flatten
  -> Dense 256 + ReLU + Dropout 0.5
  -> Dense 10 + Softmax
```

**Parameters:** ~391K

---

## Project Structure

```
Image_Classifier/
├── src/
│   ├── model.py          # CNN architecture (FashionCNN)
│   ├── train.py          # Training pipeline
│   └── predict.py        # Load model + predict
├── models/
│   ├── fashion_best.pt   # Best model (82.46%)
│   └── fashion_final.pt  # Final model
├── app/app.py            # Streamlit web app
├── reports/              # Training history plot
├── sample_images/        # Test images from dataset
├── requirements.txt
└── README.md
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **82.46%** |
| Training Time | ~2-3 min (CPU) |
| Model Size | ~1.5 MB |
| Inference Time | < 1 second / image |

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.14 | Core language |
| PyTorch 2.12 | Deep learning framework |
| torchvision | Dataset + image transforms |
| Streamlit | Web app deployment |
| matplotlib | Training history plot |
| Pillow | Image processing |

---

## Quick Start

```bash
# Clone
git clone https://github.com/daslbv/fashion-mnist-classifier.git
cd fashion-mnist-classifier

# Setup
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Train
python src/train.py

# Run web app
streamlit run app/app.py
```

Opens at http://localhost:8501. Upload an image of a clothing item to get a prediction with confidence scores.

---

## Sample Images

10 test images are included in `sample_images/` for quick testing:

`Ankle boot.png`, `Bag.png`, `Coat.png`, `Dress.png`, `Pullover.png`, `Sandal.png`, `Shirt.png`, `Sneaker.png`, `T-shirt-top.png`, `Trouser.png`

---

## Future Improvements

- Train on full 60K dataset for higher accuracy
- Add data augmentation (rotation, flip, crop)
- Experiment with deeper architectures (ResNet, VGG)
- Deploy to HuggingFace Spaces
- Add webcam real-time prediction
- Support color image classification

---

## License

MIT
