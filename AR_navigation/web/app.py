from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import time

from ai_model import predict_frame
from realtime_nav import RealtimeNavigator
from navigation import navigate

app = Flask(__name__)

navigator = RealtimeNavigator()

def parse_class(pred_class):

    zone, direction = pred_class.split("_")

    return int(zone), direction


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image_data = data["image"]

    destination = data["destination"]


    if "," in image_data:

        _, encoded = image_data.split(",",1)

    image_bytes = base64.b64decode(encoded)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    pred_class, confidence = predict_frame(image)


    stabilized_state = navigator.update(pred_class, confidence)


    if stabilized_state is None:

        return jsonify({"instruction":"대기"})


    instruction = navigate(stabilized_state, destination)


    zone, direction = parse_class(stabilized_state)


    return jsonify({

        "zone":zone,

        "direction":direction,

        "confidence":round(confidence,3),

        "instruction":instruction

    })


if __name__ == "__main__":

    app.run(host="0.0.0.0",port=5000,debug=True)