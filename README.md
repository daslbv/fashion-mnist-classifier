# Fashion-MNIST Classifier

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)

Classify fashion items (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot) using a CNN built with PyTorch.

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
**Training data:** 1,000 images (100 per class)

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **82.46%** |
| Training Time | ~2-3 min (CPU) |
| Model Size | ~1.5 MB |

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# Train (downloads Fashion-MNIST, trains on 1000 samples)
python src/train.py

# Run web app
streamlit run app/app.py
```

## Project Structure

```
Image_Classifier/
├── src/
│   ├── model.py       # CNN architecture (FashionCNN)
│   ├── train.py       # Training pipeline
│   └── predict.py     # Prediction utility
├── models/
│   ├── fashion_best.pt    # Best model (82.46%)
│   └── fashion_final.pt   # Final model
├── app/app.py         # Streamlit web app
├── reports/           # Training history plot
└── requirements.txt
```

## License

MIT
