import argparse
import math
import os
import time

import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from efficient_vit.demo_sam_model import draw_scatter, draw_binary_mask, cat_images
from efficient_vit.efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficient_vit.efficientvit.models.utils import build_kwargs_from_config
from efficient_vit.efficientvit.sam_model_zoo import create_sam_model

from proxylessnas.proxyless_gaze.deployment.onnx.main import *

def draw_gaze(image_in, arrow_head, arrow_tail, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,0,0)
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    cv2.arrowedLine(imagnew.il, 2, (255, 0, 0), thickness=5)
    
    cv2.putText(image_out, f"head: {arrow_head[0]},{arrow_head[1]}", (10,300), font, 0.8, color, 2)
    cv2.putText(image_out, f"tail: {arrow_tail[0]},{arrow_tail[1]}", (10,330), font, 0.8, color, 2)



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
        # img = segment_point(img, focus_point)
        draw_gaze(img, head, focus_point)
        get_pixels_on_line(img, head, focus_point)
    else:
        raise NotImplementedError

    return img

def all_visualize(img, gaze_pitchyaw):
    # masks = efficientvit_mask_generator.generate(img)
    plt.figure(figsize=(20, 20))
    plt.imshow(raw_image)
    # show_anns(masks)
    plt.axis("off")
    plt.savefig(f"vis_{time.time()}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)


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
    while 0<=x<H and 0<=y<W:
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
    #     cv2.circle(img, (x,y), 3, (255, 0, 0), thickness=5)


if __name__ == '__main__':
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

    provider = "CPUExecutionProvider"
    
    face_detection_session = onnxruntime.InferenceSession("./proxylessnas/proxyless_gaze/deployment/onnx/models/face_detection.onnx", providers = [provider])
    landmark_detection_session = onnxruntime.InferenceSession("./proxylessnas/proxyless_gaze/deployment/onnx/models/landmark_detection.onnx", providers = [provider])
    gaze_estimation_session = onnxruntime.InferenceSession("./proxylessnas/proxyless_gaze/deployment/onnx/models/gaze_estimation.onnx", providers = [provider])
    
    efficientvit_sam = create_sam_model(args.model, True, args.weight_url).eval()

    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam, **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator))

    cap = cv2.VideoCapture(0)
    timer = Timer()

    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    
    if args.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
    
    cnt = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        timer.start_record("whole_pipeline")
        img = frame.copy()
        
        CURRENT_TIMESTAMP = timer.get_current_timestamp()
        cnt += 1
        if cnt % 2 == 1:
            faces = detect_face(frame, face_detection_session, timer=timer)
        
        if faces is not None:
            face = faces[0]
            x1, y1, x2, y2 = face[:4]
            [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
            face = np.array([x1,y1,x2,y2,face[-1]])
            landmark, landmark_on_cropped, cropped = detect_landmark(frame, face, landmark_detection_session, timer=timer)
            landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
            gaze_pitchyaw, rvec, tvec = estimate_gaze(frame, landmark, gaze_estimation_session, timer=timer)
            gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
            
            timer.start_record("visualize")
            if args.mode == "point":
                img = point_visualize(img, gaze_pitchyaw)
            elif args.mode == "all":
                img = all_visualize(img, gaze_pitchyaw)

            timer.end_record("visualize")
        
        timer.end_record("whole_pipeline")
        img = timer.print_on_image(img)
        
        if args.save_video is not None:
            writer.write(img)
        
        cv2.imshow("onnx_demo", img)
        end = time.time()
        print("diff:", end - start)
        code = cv2.waitKey(1)
        
        if code == 27:
            break

        

