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

ZONE_GRAPH = {
    1: {2: "S"},
    2: {1: "N", 3: "S", 7: "E"},
    3: {2: "N", 4: "S", 5: "E"},
    4: {3: "N"},
    5: {3: "W", 6: "N"},
    6: {5: "S", 7: "N"},
    7: {6: "S", 2: "W"},
}


DESTINATION_TO_ZONE = {
"1강의실":4,
"2강의실":3,
"3강의실":3,
"4강의실":2,
"5강의실":1,
"6강의실":1,
"7강의실":1,
"사무실":1,
"1회의실":2,
"2회의실":3,
"3회의실":3,
"4회의실":3,
"5회의실":4,
"6회의실":4,
"7회의실":4,
"대회의실":4,
"엘리베이터":6,
"화장실":5
}