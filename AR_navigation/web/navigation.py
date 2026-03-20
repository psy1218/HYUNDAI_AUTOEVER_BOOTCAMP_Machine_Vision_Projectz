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

from zone_graph import ZONE_GRAPH, DESTINATION_TO_ZONE

TURN_MAP = {

("N","N"):"직진",
("N","E"):"우회전",
("N","S"):"후진",
("N","W"):"좌회전",

("E","N"):"좌회전",
("E","E"):"직진",
("E","S"):"우회전",
("E","W"):"후진",

("S","N"):"후진",
("S","E"):"좌회전",
("S","S"):"직진",
("S","W"):"우회전",

("W","N"):"우회전",
("W","E"):"후진",
("W","S"):"좌회전",
("W","W"):"직진",

}


def parse_current_state(state):

    zone, heading = state.split("_")

    return int(zone), heading


def navigate(current_state, destination):

    current_zone, current_heading = parse_current_state(current_state)

    destination_zone = DESTINATION_TO_ZONE[destination]

    if current_zone == destination_zone:

        return "목적지 주변에 도착"


    next_zone = list(ZONE_GRAPH[current_zone].keys())[0]

    target_direction = ZONE_GRAPH[current_zone][next_zone]

    instruction = TURN_MAP[(current_heading,target_direction)]

    return instruction