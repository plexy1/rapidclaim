import torch
from torchvision import transforms, models
from PIL import Image
import os
import json


def _build_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid()
    )
    return model


def preprocess_image(image_path):
    """Preprocess image for the model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict(image_path, model_path=None, device=None):
    """Make prediction for a single image using saved state_dict (.pth)"""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_path is None:
            model_path = os.path.join('backend', 'model', 'car_crash_model.pth')

        if not os.path.exists(model_path):
            return {'status': 'error', 'error_message': f'Model file not found: {model_path}'}

        model = _build_model()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        image_tensor = preprocess_image(image_path).to(device)

        with torch.no_grad():
            output = model(image_tensor).squeeze()
            probability = float(output.item())
            prediction = 1 if probability > 0.5 else 0

        result = {
            'crash_detected': bool(prediction),
            'confidence': probability,
            'status': 'success'
        }

    except Exception as e:
        result = {
            'status': 'error',
            'error_message': str(e)
        }

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(json.dumps(predict(image_path)))
    else:
        print(json.dumps({'status': 'error', 'error_message': 'Please provide image path as arg'}))
