import argparse
import math
import os
import time

import pickle

import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import yaml

from matplotlib.patches import Rectangle
from PIL import Image
from super_gradients.training import models
from super_gradients.common.object_names import Models

from efficient_vit.demo_sam_model import draw_scatter, draw_binary_mask, cat_images, load_image, show_anns
from efficient_vit.efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficient_vit.efficientvit.models.utils import build_kwargs_from_config
from efficient_vit.efficientvit.sam_model_zoo import create_sam_model

from proxylessnas.proxyless_gaze.deployment.onnx.main import *

def draw_gaze(image_in, arrow_head, arrow_tail, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    cv2.arrowedLine(image_out, 
                    arrow_head,
                    arrow_tail, 
                    color,
                    thickness, 
                    cv2.LINE_AA, 
                    tipLength=0.2)

    cv2.circle(image_out, arrow_tail, 2, (255, 0, 0), thickness=5)


def get_gaze_focus(img, eye_pos, pitchyaw, length):
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])

    arrow_head = tuple(np.round(eye_pos).astype(np.int32))
    arrow_tail = tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int))

    return arrow_head, arrow_tail

def segment_point(img, focus_point):
    H, W, _ = img.shape

    tmp_file = f".tmp_{time.time()}.png"
    point_coords = [focus_point]
    point_labels = [1]

    efficientvit_sam_predictor.set_image(img)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=np.array(point_coords),
        point_labels=np.array(point_labels),
        multimask_output=args.multimask,
    )

    plots = [
        draw_scatter(
            draw_binary_mask(img, binary_mask, (255, 0, 0)),
            point_coords,
            color=["g" if l == 1 else "r" for l in point_labels],
            s=10,
            ew=0.25,
            tmp_name=tmp_file,
        )
        for binary_mask in masks
    ]

    plots = cat_images(plots, axis=1)
    return plots


def point_visualize(img, gaze_pitchyaw):
    if gaze_pitchyaw is not None:
        eye_pos = landmark[-2:].mean(0)
        head, focus_point = get_gaze_focus(img, eye_pos, gaze_pitchyaw, length=300)
        img = segment_point(img, focus_point)
        draw_gaze(img, head, focus_point)
    else:
        raise NotImplementedError

    return img

def find_edge_intersection(w, h, start, end):
    x1, y1 = start
    x, y = end

    up = y <= y1
    right = x >= x1

    if abs(x) == abs(x1): # vertical case
        if y < y1:
            return (x, 0)
        elif y == y1:
            return start  # looking direct at ya
        else:
            return (x, h-1)

    m = (y-y1) / (x-x1)
    abs_m = abs(m)

    avg_slope = h/w

    if up and right: # Q1
        new_x, new_y = w-1, 0
    elif up and not right: #Q2
        new_x, new_y = 0, 0
    elif not up and right:
        new_x, new_y = w-1, h-1
    elif not up and not right:
        new_x, new_y = 0, h-1
    else:
        print("ya screwed up")
        raise Exception("didn't find edge point")

    if abs_m < avg_slope: # flat slope, will intersect with left, right edge. find y
        # eq: m * (x-start_x) + start_y
        new_y = m * (new_x - x) + y
    if abs_m > avg_slope: # tall slope, intersect with floor/ ceil. find x
        # eq: 1/m * (y-start_y) + start_x
        new_x = 1/m * (new_y - y) + x
    
    return int(new_x), int(new_y)
    

def get_pixels_on_line(img, start_point, end_point):
    """
    Get all pixel coordinates that lie on a line between start_point and end_point.

    Parameters:
    - img_shape: Tuple (height, width) representing the image dimensions.
    - start_point: Tuple (x, y) representing the starting point of the line.
    - end_point: Tuple (x, y) representing the ending point of the line.

    Returns:
    - List of (x, y) tuples representing the pixel coordinates on the line.
    """
    H, W = img.shape[:2]
    x1, y1 = start_point
    x2, y2 = find_edge_intersection(W, H, start_point, end_point)

    print("H:", H, "W:", W)

    print("edge intersection:", x2, y2)

    # Calculate differences and absolute differences between points
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine the direction along the x and y axes
    if x1 < x2:
        x_increment = 1
    else:
        x_increment = -1

    if y1 < y2:
        y_increment = 1
    else:
        y_increment = -1

    # Initialize error values and the current position
    error = dx - dy
    x = x1
    y = y1

    x_vals, y_vals = [], []

    # Iterate through the line and add pixels to the result
    # while 0<=x<H and 0<=y<W:
    while 0<=x<W and 0<=y<H:
        # Add the current pixel to the list
        x_vals.append(x)
        y_vals.append(y)

        # Calculate error and next position
        double_error = 2 * error
        if double_error > -dy:
            error -= dy
            x += x_increment
        if double_error < dx:
            error += dx
            y += y_increment

    # Add the last pixel (end_point) to the list
    x_vals.append(x2)
    y_vals.append(y2)

    # for x, y in zip(x_vals, y_vals):
    #     cv2.circle(img, (x,y), 1, (255, 0, 0), thickness=1)

    mask = np.zeros((H, W))
    mask[y_vals, x_vals] = 1

    return mask.astype(bool)

def intersects_bb(mask, bbs): # get the largest percentage overlap that this mask has with any of the bounding boxes
    best_percentage = 0

    for bb in bbs:
        x1, y1, x2, y2 = bb
        roi = mask[y1:y2+1, x1:x2+1]
        
        intersection_area = np.count_nonzero(roi)
        bb_area = abs((x2-x1+1) * (y2-y1+1))
        percentage_overlap = intersection_area / bb_area * 100
        best_percentage = max(best_percentage, percentage_overlap)
        
    return best_percentage

def check_self(mask, eye_loc):
    rev = (eye_loc[1], eye_loc[0])
    return mask[rev] == 1

def show_anns(anns, line_mask, bbs, center_pix) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)


    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    cv2.circle(img, (0,0), 5, (0, 255, 0), thickness=20)

    percentage_to_mask = {}
    for i, ann in enumerate(sorted_anns):
        m = ann["segmentation"]
        if check_self(m, center_pix): # dont want yourself
            continue
        intersection = np.logical_and(m, line_mask)
        ix = np.argwhere(intersection)
        if len(ix) > 0: # object crosses line of sight
            percentage_intersection = intersects_bb(m, bbs)
            percentage_to_mask[percentage_intersection] = (i, ix[0]) # v low chance of same thing, in this case, j replace
                
    color_mask = np.concatenate([[255, 0, 0], [0.35]])

    if len(percentage_to_mask) == 0:
        print("EMPTY PERCENTAGE TO MASK???")
    
    else:
        # for p in percentage_to_mask:
        #     max_percentage = max(percentage_to_mask)
        #     print("max percentage_overlap:", max_percentage, percentage_to_mask)
        #     mask_ix, point = percentage_to_mask[max_percentage]
        #     mask_ix, point = percentage_to_mask[p]
        #     mask = sorted_anns[mask_ix]['segmentation']
        #     img[mask] = np.concatenate([np.random.random(3), [0.35]])

        max_percentage = max(percentage_to_mask)
        print("max percentage_overlap:", max_percentage, percentage_to_mask)
        mask_ix, point = percentage_to_mask[max_percentage]
        mask = sorted_anns[mask_ix]['segmentation']
        img[mask] = color_mask
        print("point:", point)
        p = (point[1], point[0])
        plt.scatter(*p, color='blue', marker='*', s=500, edgecolors="white", linewidths=1)

    ax = plt.gca()
    ax.set_autoscale_on(False) 
    
    img[line_mask] = np.concatenate([[0, 255, 0], [0.5]])

    print(f"num masks: {len(sorted_anns)}")
    print("img.shape:", img.shape)

    ax.imshow(img) # this is important to keep the segmentations on the img
    
def all_visualize(img, start_point, end_point):
    img_out = img
    if len(img_out.shape) == 2 or img_out.shape[2] == 1:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    bb_time = time.time()
    bb_res = bb_model.predict(img)

    print("time to run bb:", time.time() - bb_time)

    eff_time = time.time()

    # masks = efficientvit_mask_generator.generate(img)

    # with open("masks/mask_wall.pkl", "wb") as f:
    #     pickle.dump(masks, f)

    with open("masks/mask_wall.pkl", 'rb') as f:
        masks = pickle.load(f)

    print("time to run eff:", time.time() - eff_time)

    bounding_box_coords = []
    for res in bb_res:
        for bounding_box in res.prediction.bboxes_xyxy:
            x1, y1, x2, y2 = [int(coord.item()) for coord in bounding_box]
            bounding_box_coords.append([x1, y1, x2, y2])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)

    line_time = time.time()
    line_mask = get_pixels_on_line(img, start_point, end_point)
    print("time to get pixels on line:", time.time() - line_time)

    plt.figure(figsize=(20, 20))
    plt.imshow(img) # after adding image cv2 stuff doesn't render

    anns = time.time()
    show_anns(masks, line_mask, bounding_box_coords, start_point)
    print("time to draw and determine masks:", time.time() - anns)
    cv2.circle(img, (0,0), 5, (0, 255, 0), thickness=20)

    plt.axis("off")

    plt.savefig(f"out/vis_{time.time()}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)

def get_bounded_point(pt, w, h):
    x, y = pt[0], pt[1]
    bounded_x = min(max(0, x), w-1)
    bounded_y = min(max(0, y), h-1)

    return (bounded_x, bounded_y)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument("--source", default="/dev/video0", type=str)
    parser.add_argument("--save-video", default=None, type=str, required=False)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--image_path", type=str, default="face.jpg")
    parser.add_argument("--output_path", type=str, default="segmented_face.png")

    parser.add_argument("--mode", type=str, default="all", choices=["point", "box", "all"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--box", type=str, default=None)

    args, opt = parser.parse_known_args()

    efficientvit_sam = create_sam_model(args.model, True, args.weight_url).eval()

    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam, **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator))

    w, h = 1280, 720
    y1 = time.time()
    bb_model = models.get(Models.YOLO_NAS_S, pretrained_weights='coco')

    # bb_model = attempt_load("yolov5m_Objects365.pt")
    y2 = time.time()
    print("time to load yolo:", y2 - y1)
    img = load_image("base_imgs/wall.png")
    img = cv2.resize(img, (w, h))

    # start_point = get_bounded_point((575, 253), w, h)
    # end_point = get_bounded_point((568, -15), w, h)
    # start_point = get_bounded_point((717, 254), w, h)
    # end_point = get_bounded_point((424, 286), w, h)
    start_point = get_bounded_point((396, 258), w, h)
    end_point = get_bounded_point((105, 237), w, h)
    print("original image size:", img.shape)
    img = all_visualize(img, start_point, end_point)

    end = time.time()

    print("total time: ", end - start)


    # bounding box  : 1.6s
    # segment       : 99.3 s
    # pixels on line: 0.0038 s
    # add good masks: 0.506 s

    # total         : 106.1 s
