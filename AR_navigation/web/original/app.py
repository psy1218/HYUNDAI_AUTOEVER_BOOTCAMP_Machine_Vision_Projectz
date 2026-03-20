from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import time

from ai_model import predict_frame
from smoothing import class_smoothing
from navigation import route_logic
from zone_graph import validate_transition

app = Flask(__name__)

frame_count = 0
location_log = []


def parse_class(pred_class):

    zone = int(pred_class.split("_")[0])
    direction = pred_class.split("_")[1]

    return zone, direction


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


@app.route("/")
def index():
    return render_template("index.html")


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


        # 모델 추론
        pred_class, confidence = predict_frame(image)

        print("model prediction:", pred_class)


        # smoothing
        smooth_class = class_smoothing(pred_class)

        print("smooth class:", smooth_class)


        # class → zone / direction
        zone, direction = parse_class(smooth_class)


        # zone validation
        final_zone = validate_transition(zone)

        print(f"zone:{zone} direction:{direction}")
        print("validated zone:", final_zone)


        log_location(final_zone, direction)


        direction_map = {
            "N":"북쪽",
            "S":"남쪽",
            "E":"동쪽",
            "W":"서쪽"
        }

        direction_kor = direction_map.get(direction, direction)


        # navigation
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


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )