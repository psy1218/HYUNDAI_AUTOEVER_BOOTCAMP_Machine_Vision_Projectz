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

from ai_model import predict_frame
from smoothing import class_smoothing
from navigation import route_logic


app = Flask(__name__)


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
# 상태 변수
# =========================================================

class_history = []
HISTORY_SIZE = 5

current_zone = None

frame_count = 0

location_log = []


# =========================================================
# zone 이동 그래프
# =========================================================

zone_graph = {
    1:[2],
    2:[1,3],
    3:[2,4],
    4:[3,5],
    5:[4,6],
    6:[5,7],
    7:[6]
}


# =========================================================
# Temporal smoothing
# =========================================================

def class_smoothing(pred_class):

    global class_history

    class_history.append(pred_class)

    if len(class_history) > HISTORY_SIZE:
        class_history.pop(0)

    final_class = max(set(class_history), key=class_history.count)

    return final_class


# =========================================================
# zone 이동 검증
# =========================================================

def validate_transition(pred_zone):

    global current_zone

    if current_zone is None:
        current_zone = pred_zone
        return pred_zone

    if pred_zone in zone_graph.get(current_zone, []):
        current_zone = pred_zone
        return pred_zone

    return current_zone


# =========================================================
# class → zone / direction 분리
# =========================================================

def parse_class(pred_class):

    zone = int(pred_class.split("_")[0])
    direction = pred_class.split("_")[1]

    return zone, direction


# =========================================================
# 위치 로그
# =========================================================

def log_location(zone, direction):

    global location_log

    timestamp = time.strftime("%H:%M:%S")

    entry = {
        "time": timestamp,
        "zone": zone,
        "direction": direction
    }

    location_log.append(entry)

    if len(location_log) > 100:
        location_log.pop(0)

    print(f"[LOCATION] {timestamp} | zone:{zone} direction:{direction}")


# =========================================================
# navigation 로직
# =========================================================

def route_logic(zone, direction, destination):

    if not destination:
        return "목적지를 선택하세요."

    return f"{destination} 방향으로 안내 중입니다."


# =========================================================
# 웹 페이지
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# 프레임 추론
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    global frame_count

    try:

        start_time = time.time()

        data = request.get_json()

        image_data = data.get("image", "")
        destination = data.get("destination", "").strip()

        if not image_data:
            return jsonify({"error": "이미지 데이터 없음"}), 400


        # base64 header 제거
        if "," in image_data:
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data


        image_bytes = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        frame = np.array(image)

        frame_count += 1

        print(f"\n===== Frame {frame_count} =====")
        print("destination:", destination)
        print("frame shape:", frame.shape)


        # =========================================================
        # AI 모델 추론
        # =========================================================

        input_tensor = transform(image)

        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(input_tensor)

            probs = torch.softmax(outputs, dim=1)

            pred = torch.argmax(probs, dim=1).item()


        pred_class = CLASS_NAMES[pred]

        confidence = float(probs[0][pred])


        print("model prediction:", pred_class)
        print("confidence:", confidence)


        # =========================================================
        # smoothing
        # =========================================================

        smooth_class = class_smoothing(pred_class)

        print("history:", class_history)
        print("smooth class:", smooth_class)


        # =========================================================
        # zone validation
        # =========================================================

        zone, direction = parse_class(smooth_class)

        final_zone = validate_transition(zone)


        print(f"zone:{zone} direction:{direction}")
        print("validated zone:", final_zone)


        log_location(final_zone, direction)


        # =========================================================
        # direction 변환
        # =========================================================

        direction_map = {
            "N":"북쪽",
            "S":"남쪽",
            "E":"동쪽",
            "W":"서쪽"
        }

        direction_kor = direction_map.get(direction, direction)


        # =========================================================
        # navigation
        # =========================================================

        instruction = route_logic(final_zone, direction_kor, destination)


        process_time = round(time.time() - start_time, 3)

        print(f"process time: {process_time:.3f}s")


        return jsonify({
            "zone": final_zone,
            "direction": direction_kor,
            "confidence": round(confidence,3),
            "instruction": instruction
        })


    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({"error": str(e)}), 500


# =========================================================
# 서버 실행
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
