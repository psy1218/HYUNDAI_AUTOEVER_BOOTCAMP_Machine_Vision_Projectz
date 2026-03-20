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

zone_graph = {
    1:[2],
    2:[1,3],
    3:[2,4],
    4:[3,5],
    5:[4,6],
    6:[5,7],
    7:[6]
}

current_zone = None


def validate_transition(pred_zone):

    global current_zone

    if current_zone is None:
        current_zone = pred_zone
        return pred_zone

    if pred_zone in zone_graph.get(current_zone, []):
        current_zone = pred_zone
        return pred_zone

    return current_zone