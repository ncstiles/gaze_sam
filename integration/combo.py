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


from trt_sam import EfficientViTSamAutomaticMaskGenerator, SamPad, SamResize
import torchvision.transforms as transforms
from segment_anything.utils.amg import MaskData
    
from efficient_vit.efficientvit.sam_model_zoo import create_sam_model

from proxylessnas.proxyless_gaze.deployment.onnx.demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from proxylessnas.proxyless_gaze.deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter

from utils_vit import *
from utils_gaze import *
from utils_yolo import * 

from load_engine import *

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l0")
    parser.add_argument("--image_path", type=str, default="../base_imgs/gum.png")
    parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")
    parser.add_argument("--gaze_start", type=str, default=f"[{717},{254}]")
    parser.add_argument("--gaze_end", type=str, default=f"[{424},{286}]")
    args, _ = parser.parse_known_args()
    return args    

def preprocess(image):
    transform = transforms.Compose(
        [
            SamResize(512),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            ),
            SamPad(512),
        ]
    )
    return transform(image).unsqueeze(dim=0).cuda()


def prime_encoder_decoder(trt_encoder_path, trt_decoder_path, image):
    encoder = load_image_encoder_engine(trt_encoder_path)
    decoder = load_mask_decoder_engine(trt_decoder_path)

    image_path = "../base_imgs/cup.png"
    image = np.array(Image.open(image_path).convert("RGB"))

    preprocessed_image = preprocess(image)
        
    for _ in range(2):
        features = encoder(preprocessed_image)

        mask_input = torch.tensor(np.zeros((1, 1, 256, 256), dtype=np.float32)).cuda()
        has_mask_input = torch.tensor(np.zeros(1, dtype=np.float32)).cuda()

        point_coords = torch.randint(low=0, high=1024, size=(32, 1, 2), dtype=torch.float).cuda()
        point_labels = torch.randint(low=0, high=4, size=(32, 1), dtype=torch.float).cuda()

        decoder(features, point_coords, point_labels, mask_input, has_mask_input)


def prime_gaze_engines(trt_face_detection, trt_landmark_detection, trt_gaze_estimation, gaze_smoother, landmark_smoother, bbox_smoother, timer):
    image_path = "../base_imgs/cup.png"
    frame = np.array(Image.open(image_path).convert("RGB"))
    faces = detect_face_trt(frame, trt_face_detection, timer)

    CURRENT_TIMESTAMP = timer.get_current_timestamp()
    if faces is not None:
        
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        
        [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
        face = np.array([x1,y1,x2,y2,face[-1]])        
        
        landmark, _, _ = detect_landmark_trt(frame, face, trt_landmark_detection, timer)
        landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
        
        gaze_pitchyaw, _, _ = estimate_gaze_trt(frame, landmark, trt_gaze_estimation, timer)        
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
    

def prime_yolo(trt_yolo):
    for i in range(3):
        image_path = "../base_imgs/psycho_out.png"
        raw_image = np.array(Image.open(image_path).convert("RGB"))
        image_yolo = cv2.resize(raw_image, (640, 640)) # must be (640, 640) to be compatible with engine
        expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
        yolo_img = torch.Tensor(expanded_img).cuda()
        trt_yolo(yolo_img)

def main(image_path, gaze_start, gaze_end):
    y = time.time()
    args = get_cli_args()

    # point processing
    args.gaze_start = yaml.safe_load(args.gaze_start)
    args.gaze_end = yaml.safe_load(args.gaze_end)

    # remove, this is just for testing
    args.image_path = image_path
    args.gaze_start = gaze_start
    args.gaze_end = gaze_end

    # vit initialization
    # trt_encoder_path = "engines/vit/encoder_k9_fp32_trt8.6.engine"
    trt_encoder_path = "engines/vit/encoder_fp16_trt8.6.engine"

    trt_decoder_path = "engines/vit/decoder_fp16_k9_unstacked_l0_opset11.engine"
    efficientvit_sam = create_sam_model(args.model, True, None).cuda().eval()
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, trt_encoder_path=trt_encoder_path, trt_decoder_path=trt_decoder_path)

    # gaze initialization
    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    

    trt_face_detection = load_face_detection_engine("engines/gaze/face_detection_fp32_k9_trt8.6.engine") 
    trt_landmark_detection = load_landmark_detection_engine("engines/gaze/landmark_detection_fp32_k9_trt8.6.engine")
    trt_gaze_estimation = load_gaze_estimation_engine("engines/gaze/gaze_estimation_fp32_k9_trt8.6.engine")

    timer = Timer()
    timer.start_record("whole_pipeline")
    CURRENT_TIMESTAMP = timer.get_current_timestamp()

    # yolo initialization
    # trt_yolo = load_yolo_engine("engines/yolo/yolo_fp32_k9_trt8.6.engine")
    trt_yolo = load_yolo_engine("engines/yolo/yolo_k9_int8_trt8.6.engine") 

    z = time.time()

    # load image
    a = time.time()
    # raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    raw_image = cv2.imread(args.image_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    b = time.time()

    if raw_image.shape[0] * raw_image.shape[1] > 1280 * 720:
        raw_image = cv2.resize(raw_image, (1280, 720))
    
    ggg = time.time()
    prime_encoder_decoder(trt_encoder_path, trt_decoder_path, raw_image)
    hhh = time.time()

    eee = time.time()
    prime_gaze_engines(trt_face_detection, trt_landmark_detection, trt_gaze_estimation, gaze_smoother, landmark_smoother, bbox_smoother, timer)
    fff = time.time()

    yyy = time.time()
    prime_yolo(trt_yolo)
    zzz = time.time()
    
    bb = time.time()

    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    # run gaze model
    frame = raw_image
    e = time.time()
    faces = detect_face_trt(frame, trt_face_detection, timer)
    f = time.time()
    # sometimes no face, want these vars to still exist
    g, h, i, j, k, l, m = 0, 0, 0, 0, 0, 0, 0
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
    
    p = time.time()
    gaze_points = get_pixels_on_line(raw_image, args.gaze_start, args.gaze_end)
    NUM_POINTS = 32
    gaze_start, gaze_end = gaze_points[0], gaze_points[-1]
    indices = np.linspace(0, len(gaze_points) - 1, NUM_POINTS, dtype=int)
    gaze_points = gaze_points[indices]

    q = time.time()
    # run vit model
    c = time.time()
    masks = efficientvit_mask_generator.generate(raw_image, gaze_points)
    d = time.time()

    # run yolo model
    qq = time.time()
    image_yolo = cv2.resize(raw_image, (640, 640)) # must be (640, 640) to be compatible with engine
    expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
    yolo_img = torch.Tensor(expanded_img).cuda()
    ss = time.time()
    predictions = trt_yolo(yolo_img)
    s = time.time()
    bounding_boxes = visualize_bounding_boxes(raw_image, predictions, raw_image.shape[:2])
    t = time.time()

    timer.start_record("visualize")
    n = time.time()
    # show_frame = visualize_simple(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
    if faces is not None:
        show_frame = visualize(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
    o = time.time()
    timer.end_record("visualize")
    timer.end_record("whole_pipeline")
    # show_frame = timer.print_on_image(show_frame)
    
    # visualize
    v = time.time()
    # raw_image = show_anns(raw_image, masks)
    raw_image = show_one_ann(masks, gaze_points, gaze_start, gaze_end, bounding_boxes, args.gaze_start, raw_image)
    
    w = time.time()
    plt.axis("off")
    plt.imshow(raw_image)
    xx = time.time()
    plt.savefig(f"{args.output_path}", format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    x = time.time()
    
    print()

    print("encoder/decoder priming run:", hhh - ggg)
    print("all gaze engines priming run:", fff - eee)
    print("yolo priming run:", zzz - yyy)
    print("load img:", b - a)
    print("resize img:", bb - b)

    print()

    print("detect face (primed):", f - e)
    print("smooth + extract face (primed):", h - g)
    print("detect landmark (primed):", j - i)
    print("smooth landmark (primed):", k - j)
    print("detect gaze (primed):", l - k)
    print("smooth gaze (primed):", m - l)
    print("visualize gaze:", o - n)
    print("get gaze mask:", q - p)
    print("create plots:", v - o)

    print()

    print("prep yolo img:", ss - qq)
    print("yolo pred:", s - ss)
    print("draw and get yolo boxes:", t - s)
    print("total yolo:", t - qq)


    print()

    print("generate masks:", d - c)
    print("generate gaze:", q - e)
    print("generate yolo:", t - qq)
    print("segment one mask:", w - v)

    print()

    print("display image:", xx - w)
    print(f"save to file ({args.output_path}):", x - w)
    print("non-load total:", w - e)
    print("load total:", z - y)

    return w-e, t - qq, s - ss

if __name__ == "__main__":
    files = ["../base_imgs/gum.png", "../base_imgs/help.png", "../base_imgs/pen.png", "../base_imgs/psycho.png", "../base_imgs/workpls_v2.png", "../base_imgs/zz.png"]
    eyes = [(717, 254), (575, 253), (617, 288), (595, 361), (746, 435), (485, 329)]
    tails = [(424, 286), (568, 0), (808, 242), (757, 396), (930, 434), (189, 362)]

    avg_total_time = 0
    avg_yolo_time = 0
    avg_yolo_pred_time = 0
    i = 0
    for file, eye, tail in zip(files, eyes, tails):
        print()
        print(f"~~~ ITER {i+1} with file {file} ~~~")
        total_time, yolo_time, yolo_pred_time =  main(file, eye, tail)
        avg_total_time += total_time
        avg_yolo_time += yolo_time
        avg_yolo_pred_time += yolo_pred_time
        i += 1
        print()

    print()
    print("yolo engine time:", avg_yolo_pred_time/len(files))
    print("total yolo time:", avg_yolo_time/len(files))
    print()
    print("average total time:", avg_total_time / len(files))

# elts = "generate masks: 0.12423539161682129",
# "detect face (primed): 0.002354860305786133",
# "smooth + extract face (primed): 5.3882598876953125e-05",
# "detect landmark (primed): 0.0012483596801757812",
# "smooth landmark (primed): 0.0006034374237060547",
# "detect gaze (primed): 0.004658937454223633",
# "smooth gaze (primed): 1.811981201171875e-05",
# "visualize gaze: 0.0008323192596435547",
# "create plots: 8.821487426757812e-06",
# "get gaze mask: 0.002750873565673828",
# "prep yolo img: 0.0014014244079589844",
# "yolo img torch: 0.005392551422119141",
# "yolo pred: 0.0019392967224121094",
# "draw and get yolo boxes: 0.008787870407104492",
# "segment one mask: 0.026651382446289062"]

# up to one mask optimization
# load img: 0.08280062675476074
# resize img: 2.0817110538482666
# generate masks: 0.06017279624938965
# detect face (primed): 0.0026962757110595703
# smooth + extract face (primed): 4.4345855712890625e-05
# detect landmark (primed): 0.0009033679962158203
# smooth landmark (primed): 0.0005803108215332031
# detect gaze (primed): 0.0034503936767578125
# smooth gaze (primed): 1.2636184692382812e-05
# visualize gaze: 0.0007472038269042969
# create plots: 5.9604644775390625e-06
# get gaze mask: 0.0003733634948730469
# prep yolo img: 0.004436969757080078
# yolo pred: 0.0030698776245117188
# draw and get yolo boxes: 0.003017902374267578
# segment one mask: 0.007338762283325195