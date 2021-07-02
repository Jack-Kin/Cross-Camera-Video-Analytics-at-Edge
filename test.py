import torch
import numpy as np
import cv2
from typing import Dict
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

PERSON_CLASS=0


def float2int(x):
    return np.round(x.item()).astype(np.int32)


def init():
    # YOLO Model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom
    yolo_model.classes = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 25, 26, 27, 28, 32, 39, 56, 60]

    return yolo_model


# Inference
def object_detect(od_model, img):
    results = od_model(img)

    # Results
    results.print()
    results.show()
    print(results.xyxy[0])
    for xmin, ymin, xmax, ymax, conf, label in results.xyxy[0].cpu():
        box = img[float2int(ymin):float2int(ymax), float2int(xmin):float2int(xmax)]
        



yolo_m = init()
im = cv2.imread('./ssd.pytorch-master/data/test1.png')[:, :, ::-1]
object_detect(yolo_m, im)
