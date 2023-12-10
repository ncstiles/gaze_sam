import argparse

import cv2
import numpy as np


import torch

import matplotlib.pyplot as plt

from load_engine import load_yolo_engine
from utils_yolo import visualize_bounding_boxes

trtpath = "engines/yolo/yolo.engine"
imgpath = "../base_imgs/wall.png"

yolo_engine = load_yolo_engine(trtpath)

raw_image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
image = cv2.resize(raw_image, (640, 640))
expanded_img = np.transpose(np.expand_dims(image, axis=0), (0, 3, 1, 2))

predictions = yolo_engine(torch.Tensor(expanded_img).cuda())

print("raw image shape:", raw_image.shape)
visualize_bounding_boxes(raw_image, predictions, raw_image.shape[:2])

plt.figure(figsize=(20, 20))
plt.imshow(raw_image) # after adding image cv2 stuff doesn't render
plt.axis("off")
plt.savefig(f"yolo_resized.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
