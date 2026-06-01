import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FashionCNN


CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path='models/fashion_best.pt'):
    if not Path(model_path).exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    model = FashionCNN(num_classes=10).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, image, top_k=3):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)

    results = [
        (CLASSES[idx], round(prob.item() * 100, 2))
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]

    return results
