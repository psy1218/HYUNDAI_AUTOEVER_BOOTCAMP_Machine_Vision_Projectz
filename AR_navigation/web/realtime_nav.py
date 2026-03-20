from collections import deque, Counter
from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
import numpy as np
import time

class RealtimeNavigator:

    def __init__(self, history_size=5, min_conf=0.5):

        self.history_size = history_size
        self.min_conf = min_conf

        self.pred_history = deque(maxlen=history_size)

        self.last_state = None


    def update(self, pred_class, confidence):

        if confidence < self.min_conf:

            return self.last_state


        self.pred_history.append(pred_class)


        voted = Counter(self.pred_history).most_common(1)[0][0]


        self.last_state = voted

        return voted