from flask import Flask, request, jsonify, render_template
import base64
import io
import time

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
import numpy as np
import cv2

# ==============================
# 설정
# ==============================

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

NUM_CLASSES = len(CLASS_NAMES)

# ==============================
# 상태 저장
# ==============================

zone_history = []
HISTORY_SIZE = 5
current_zone = None
frame_count = 0

# ==============================
# 이동 그래프 (수정 필요)
# ==============================

zone_graph = {
    1:[2],
    2:[1,3],
    3:[2,4],
    4:[3,5],
    5:[4,6],
    6:[5],
    7:[6]
}

# ==============================
# 모델 생성
# ==============================

def build_model(num_classes=28):

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ==============================
# 모델 로드
# ==============================

def load_best_model(model_path, device):

    model = build_model(NUM_CLASSES)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        else:
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


print("Loading model...")
model = load_best_model(MODEL_PATH, DEVICE)
print("Model loaded")


# ==============================
# 이미지 전처리
# ==============================

transform = transforms.Compose([
    transforms.Resize((256,384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==============================
# temporal smoothing
# ==============================

def temporal_smoothing(pred_zone):

    global zone_history

    zone_history.append(pred_zone)

    if len(zone_history) > HISTORY_SIZE:
        zone_history.pop(0)

    final_zone = max(set(zone_history), key=zone_history.count)

    return final_zone


# ==============================
# 이동 검증
# ==============================

def validate_transition(pred_zone):

    global current_zone

    if current_zone is None:
        current_zone = pred_zone
        return pred_zone

    if pred_zone in zone_graph.get(current_zone, []):
        current_zone = pred_zone
        return pred_zone

    return current_zone


# ==============================
# 방향 파싱
# ==============================

def parse_prediction(pred_class):

    zone = int(pred_class.split("_")[0])
    direction = pred_class.split("_")[1]

    return zone, direction


# ==============================
# 길 안내 로직
# ==============================

def route_logic(zone, direction, destination):

    if not destination:
        return "목적지를 선택하세요."

    return f"{destination} 방향으로 이동하세요"


# ==============================
# Flask
# ==============================

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    global frame_count

    try:

        start_time = time.time()

        data = request.get_json()

        image_data = data.get("image","")
        destination = data.get("destination","")

        if "," in image_data:
            _, encoded = image_data.split(",",1)
        else:
            encoded = image_data

        image_bytes = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        frame = np.array(image)

        frame_count += 1

        print("\n===== Frame",frame_count,"=====")

        # ==============================
        # 모델 추론
        # ==============================

        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(input_tensor)

            probs = torch.softmax(outputs,dim=1)

            confidence, pred = torch.max(probs,1)

        pred_idx = pred.item()
        confidence = float(confidence.item())

        pred_class = CLASS_NAMES[pred_idx]

        print("pred class:",pred_class)

        zone, direction = parse_prediction(pred_class)

        print("zone:",zone,"direction:",direction)

        # ==============================
        # smoothing
        # ==============================

        smooth_zone = temporal_smoothing(zone)

        final_zone = validate_transition(smooth_zone)

        print("final zone:",final_zone)

        instruction = route_logic(final_zone,direction,destination)

        process_time = round(time.time()-start_time,3)

        print("process time:",process_time)

        return jsonify({
            "zone": final_zone,
            "direction": direction,
            "confidence": confidence,
            "instruction": instruction
        })

    except Exception as e:

        print("ERROR:",e)

        return jsonify({"error":str(e)}),500


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)