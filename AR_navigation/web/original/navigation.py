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

def route_logic(zone, direction, destination):

    if not destination:
        return "목적지를 선택하세요."

    return f"{destination} 방향으로 안내 중입니다."