import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- Model Definitions --------------------

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class ResNetModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        model = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.attention = AttentionModule(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.demo_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.6))
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2 + 32, 512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, left_img, right_img, demo):
        left_features = self.backbone(left_img)
        right_features = self.backbone(right_img)
        left_features = self.attention(left_features)
        right_features = self.attention(right_features)
        left_features = self.pool(left_features).flatten(1)
        right_features = self.pool(right_features).flatten(1)
        demo_features = self.demo_fc(demo)
        combined = torch.cat((left_features, right_features, demo_features), dim=1)
        return self.classifier(combined)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        model = models.efficientnet_b0(pretrained=False)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.attention = AttentionModule(1280)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.demo_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.6))
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 2 + 32, 512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, left_img, right_img, demo):
        left_features = self.backbone(left_img)
        right_features = self.backbone(right_img)
        left_features = self.attention(left_features)
        right_features = self.attention(right_features)
        left_features = self.pool(left_features).flatten(1)
        right_features = self.pool(right_features).flatten(1)
        demo_features = self.demo_fc(demo)
        combined = torch.cat((left_features, right_features, demo_features), dim=1)
        return self.classifier(combined)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        model = models.densenet121(pretrained=False)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.attention = AttentionModule(1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.demo_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Dropout(0.6))
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2 + 32, 512), nn.ReLU(), nn.Dropout(0.7),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, left_img, right_img, demo):
        left_features = self.backbone(left_img)
        right_features = self.backbone(right_img)
        left_features = self.attention(left_features)
        right_features = self.attention(right_features)
        left_features = self.pool(left_features).flatten(1)
        right_features = self.pool(right_features).flatten(1)
        demo_features = self.demo_fc(demo)
        combined = torch.cat((left_features, right_features, demo_features), dim=1)
        return self.classifier(combined)

class EnsembleModel(nn.Module):
    def __init__(self, models, weights):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)

    def forward(self, left_img, right_img, demo):
        outputs = [torch.sigmoid(model(left_img, right_img, demo)) for model in self.models]
        outputs = torch.stack(outputs)
        return torch.sum(outputs * self.weights[:, None, None], dim=0)

# -------------------- Load Model --------------------

def load_ensemble_model(model_paths, weights):
    resnet_model = ResNetModel().to(device)
    efficientnet_model = EfficientNetModel().to(device)
    densenet_model = DenseNetModel().to(device)

    resnet_model.load_state_dict(torch.load(model_paths[0], map_location=device))
    efficientnet_model.load_state_dict(torch.load(model_paths[1], map_location=device))
    densenet_model.load_state_dict(torch.load(model_paths[2], map_location=device))

    models = [resnet_model, efficientnet_model, densenet_model]
    ensemble = EnsembleModel(models, weights).to(device)
    ensemble.eval()
    return ensemble

# -------------------- Prediction --------------------

def predict_disease(left_img_path, right_img_path, model_dir, age=50, gender="Male", thresholds=None):
    if thresholds is None:
        thresholds = {
            'N': 0.5, 'D': 0.6, 'G': 0.5,
            'C': 0.5, 'A': 0.65, 'H': 0.7,
            'M': 0.5, 'O': 0.5
        }

    class_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    class_full_names = {
        'N': 'Normal',
        'D': 'Diabetic Retinopathy',
        'G': 'Glaucoma',
        'C': 'Cataract',
        'A': 'Age-related Macular Degeneration',
        'H': 'Hypertensive Retinopathy',
        'M': 'Myopia',
        'O': 'Other'
    }

    model_paths = [
        os.path.join(model_dir, "resnet50_best.pth"),
        os.path.join(model_dir, "efficientnet_best.pth"),
        os.path.join(model_dir, "densenet_best.pth")
    ]
    weights = [0.33, 0.33, 0.34]  # Adjust if needed

    try:
        ensemble = load_ensemble_model(model_paths, weights)
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    gender_value = 1 if gender.lower() == "male" else 0
    demo_tensor = torch.tensor([[age / 100.0, gender_value]], dtype=torch.float32).to(device)

    try:
        left_img = np.array(Image.open(left_img_path).convert('RGB'))
        right_img = np.array(Image.open(right_img_path).convert('RGB'))

        left_img = transform(image=left_img)['image'].unsqueeze(0).to(device)
        right_img = transform(image=right_img)['image'].unsqueeze(0).to(device)
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

    with torch.no_grad():
        with autocast():
            outputs = ensemble(left_img, right_img, demo_tensor)

    predictions = []
    probabilities = {}
    for i, class_name in enumerate(class_names):
        prob = outputs[0, i].item()
        probabilities[class_name] = prob
        if prob >= thresholds[class_name]:
            predictions.append(class_name)

    if not predictions and len(probabilities) > 0:
        max_class = max(probabilities, key=probabilities.get)
        predictions.append(max_class)

    result = {
        'predictions': predictions,
        'probabilities': probabilities,
        'is_normal': 'N' in predictions and len(predictions) == 1,
        'diagnosis': []
    }

    if result['is_normal']:
        result['diagnosis'].append("Normal eye condition detected.")
    else:
        if 'N' in predictions:
            predictions.remove('N')
        for disease in predictions:
            result['diagnosis'].append(f"{class_full_names[disease]} detected with {probabilities[disease]*100:.1f}% confidence.")

    return result

# -------------------- CLI Entry Point --------------------

def main():
    parser = argparse.ArgumentParser(description="Fundus image disease prediction")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory with model files')
    parser.add_argument('--left_img_path', type=str, required=True, help='Path to left eye image')
    parser.add_argument('--right_img_path', type=str, required=True, help='Path to right eye image')
    parser.add_argument('--age', type=int, default=60, help='Patient age')
    parser.add_argument('--gender', type=str, default="Male", help='Gender: Male or Female')
    args = parser.parse_args()

    thresholds = {
        'N': 0.5, 'D': 0.65, 'G': 0.45,
        'C': 0.5, 'A': 0.7, 'H': 0.7,
        'M': 0.5, 'O': 0.5
    }

    print("\nRunning fundus image disease prediction...")
    print(f"Left Image: {os.path.basename(args.left_img_path)}")
    print(f"Right Image: {os.path.basename(args.right_img_path)}")
    print(f"Patient Info: Age {args.age}, Gender {args.gender}")

    result = predict_disease(
        args.left_img_path, args.right_img_path, args.model_dir,
        age=args.age, gender=args.gender, thresholds=thresholds
    )

    if result:
        print("\n----- RESULT -----")
        if result['is_normal']:
            print("✅ Eyes are healthy. No signs of disease.")
        else:
            print("🔍 Detected Diseases:")
            for diag in result['diagnosis']:
                print(f"  - {diag}")
        print("\nProbabilities:")
        for disease, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {disease}: {prob*100:.1f}%")
    else:
        print("❌ Prediction failed. Check image paths and model files.")

if __name__ == "__main__":
    main()
