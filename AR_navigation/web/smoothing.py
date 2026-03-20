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

class_history = []
HISTORY_SIZE = 5


def class_smoothing(pred_class):

    global class_history

    class_history.append(pred_class)

    if len(class_history) > HISTORY_SIZE:
        class_history.pop(0)

    final_class = max(set(class_history), key=class_history.count)

    return final_class