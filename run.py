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

def segment_image(img, focus_point):
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


def visualize(img, gaze_pitchyaw):
    if gaze_pitchyaw is not None:
        eye_pos = landmark[-2:].mean(0)
        head, focus_point = get_gaze_focus(img, eye_pos, gaze_pitchyaw, length=300)
        img = segment_image(img, focus_point)
        draw_gaze(img, head, focus_point)
    else:
        print("no pitchyaw")

    return img

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
        show_frame = frame.copy()
        
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
            show_frame = visualize(show_frame, gaze_pitchyaw)
            timer.end_record("visualize")
        
        timer.end_record("whole_pipeline")
        show_frame = timer.print_on_image(show_frame)
        
        if args.save_video is not None:
            writer.write(show_frame)
        
        cv2.imshow("onnx_demo", show_frame)
        end = time.time()
        print("diff:", end - start)
        code = cv2.waitKey(1)
        
        if code == 27:
            break

        

