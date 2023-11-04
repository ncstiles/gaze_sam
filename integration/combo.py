# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from typing import Any, Union, Tuple, Optional, List, Dict


from trt_sam import EfficientViTSamAutomaticMaskGenerator
from efficient_vit.efficientvit.sam_model_zoo import create_sam_model

from proxylessnas.proxyless_gaze.deployment.onnx.demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from proxylessnas.proxyless_gaze.deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter

from utils_vit import *
from utils_gaze import *

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l1")
    # parser.add_argument("--image_path", type=str, default="../base_imgs/fig/cat.jpg")
    parser.add_argument("--image_path", type=str, default="../base_imgs/workpls.png")
    parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_cli_args()

    # vit initialization
    trt_encoder_path = "engines/vit/encoder.engine"
    trt_decoder_path = "engines/vit/decoder.engine"
    efficientvit_sam = create_sam_model(args.model, True, None).cuda().eval()
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, trt_encoder_path=trt_encoder_path, trt_decoder_path=trt_decoder_path)

    # gaze initialization
    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    
    trt_face_detection = load_face_detection_engine("engines/gaze/face_detection.engine")
    trt_landmark_detection = load_landmark_detection_engine("engines/gaze/landmark_detection.engine")
    trt_gaze_estimation = load_gaze_estimation_engine("engines/gaze/gaze_estimation.engine")
    
    timer = Timer()
    timer.start_record("whole_pipeline")
    CURRENT_TIMESTAMP = timer.get_current_timestamp()


    # load image
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    if raw_image.shape[0] * raw_image.shape[1] > 1280 * 720:
        raw_image = cv2.resize(raw_image, (1280, 720))

    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    start = time.time()
    masks = efficientvit_mask_generator.generate(raw_image)
    end = time.time()

    frame = raw_image
    faces = detect_face_trt(frame, trt_face_detection, timer)
    if faces is not None:
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
        face = np.array([x1,y1,x2,y2,face[-1]])
        
        landmark, _, _ = detect_landmark_trt(frame, face, trt_landmark_detection, timer)
        landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
        
        gaze_pitchyaw, rvec, tvec = estimate_gaze_trt(frame, landmark, trt_gaze_estimation, timer)
        
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
        timer.start_record("visualize")
        show_frame = visualize(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
        timer.end_record("visualize")

    timer.end_record("whole_pipeline")
    show_frame = timer.print_on_image(show_frame)
    
    print("mask generation time:", end - start)
    plt.figure(figsize=(20, 20))
    plt.imshow(raw_image)
    show_anns(masks)
    plt.axis("off")

    print(f"saving img to {args.output_path}")
    plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    main()