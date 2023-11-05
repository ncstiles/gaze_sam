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
from utils_yolo import * 

from load_engine import *

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l1")
    # parser.add_argument("--image_path", type=str, default="../base_imgs/fig/cat.jpg")
    parser.add_argument("--image_path", type=str, default="../base_imgs/workpls.png")
    parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")
    parser.add_argument("--gaze_start", type=str, default=f"[{746},{435}]")
    parser.add_argument("--gaze_end", type=str, default=f"[{930},{434}]")
    args, _ = parser.parse_known_args()
    return args


def main():
    y = time.time()
    args = get_cli_args()

    # point processing
    args.gaze_start = yaml.safe_load(args.gaze_start)
    args.gaze_end = yaml.safe_load(args.gaze_end)

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

    # yolo initialization
    trt_yolo = load_yolo_engine("engines/yolo/yolo.engine")

    z = time.time()

    # load image
    a = time.time()
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    if raw_image.shape[0] * raw_image.shape[1] > 1280 * 720:
        raw_image = cv2.resize(raw_image, (1280, 720))
    
    b = time.time()

    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")
    
    # run gaze model
    frame = raw_image
    e = time.time()
    faces = detect_face_trt(frame, trt_face_detection, timer)
    f = time.time()
    if faces is not None:
        g = time.time()
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
        face = np.array([x1,y1,x2,y2,face[-1]])
        h = time.time()
        
        i = time.time()
        landmark, _, _ = detect_landmark_trt(frame, face, trt_landmark_detection, timer)
        j = time.time()
        landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
        k = time.time()
        
        gaze_pitchyaw, rvec, tvec = estimate_gaze_trt(frame, landmark, trt_gaze_estimation, timer)
        l = time.time()
        
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
        m = time.time()
        timer.start_record("visualize")
        n = time.time()
        show_frame = visualize_simple(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
        o = time.time()
        timer.end_record("visualize")

        timer.end_record("whole_pipeline")
        # show_frame = timer.print_on_image(show_frame)
    
    p = time.time()
    gaze_points, gaze_mask = get_pixels_on_line(raw_image, args.gaze_start, args.gaze_end)
    q = time.time()

    # run vit model
    c = time.time()
    masks = efficientvit_mask_generator.generate(raw_image, gaze_points)
    d = time.time()

    # run yolo model
    image_yolo = cv2.resize(raw_image, (640, 640)) # must be (640, 640) to be compatible with engine
    expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
    r = time.time()
    predictions = trt_yolo(torch.Tensor(expanded_img).cuda())
    s = time.time()
    visualize_bounding_boxes(raw_image, predictions, raw_image.shape[:2])
    t = time.time()
    bounding_boxes = get_bounding_boxes(predictions, raw_image.shape[:2])
    u = time.time()

    # visualize
    plt.figure(figsize=(20, 20))
    plt.imshow(raw_image)
    v = time.time()
    # show_anns(masks)
    show_one_ann(masks, gaze_mask, bounding_boxes, args.gaze_start)
    
    w = time.time()
    plt.axis("off")
    plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    x = time.time()

    print("full without load time:", time.time() - a)
    print(f"saving img to {args.output_path}")
    
    print()

    print("load and resize img:", b - a)
    print("generate masks:", d - c)
    print("detect face:", f - e)
    print("smooth + extract face:", h - g)
    print("detect landmark:", j - i)
    print("smooth landmark:", k - j)
    print("detect gaze:", l - k)
    print("smooth gaze:", m - l)
    print("visualize gaze:", o - n)
    print("get gaze mask:", q - p)
    print("prep yolo img:", r - q)
    print("yolo pred:", s - r)
    print("visualize yolo:", t - s)
    print("get bounding boxes:", u - t)
    print("show non-mask img:", v - u)
    print("segment one mask:", w - v)

    print()
    
    print("save to file:", x - w)
    print("non-load total:", x - a)
    print("load total:", z - y)

if __name__ == "__main__":
    main()