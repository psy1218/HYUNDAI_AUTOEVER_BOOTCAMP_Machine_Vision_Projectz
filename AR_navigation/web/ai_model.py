from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import cv2
import numpy as np
import time

import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================================================
# 모델 설정
# =========================================================

MODEL_PATH = "./exp3_bigimg_simple_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "1_E","1_N","1_S","1_W",
    "2_E","2_N","2_S","2_W",
    "3_E","3_N","3_S","3_W",
    "4_E","4_N","4_S","4_W",
    "5_E","5_N","5_S","5_W",
    "6_E","6_N","6_S","6_W",
    "7_E","7_N","7_S","7_W"
]


# =========================================================
# 모델 생성
# =========================================================

def build_model():

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 28)

    return model


# =========================================================
# 모델 로드
# =========================================================

print("Loading model...")

model = build_model()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully")


# =========================================================
# 이미지 transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((256,384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


 # =========================================================
# AI 모델 추론
# =========================================================
def predict_frame(image):

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)

        pred = torch.argmax(probs, dim=1).item()

    pred_class = CLASS_NAMES[pred]

    confidence = float(probs[0][pred])

    return pred_class, confidence