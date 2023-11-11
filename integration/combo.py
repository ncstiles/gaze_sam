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
    # parser.add_argument("--image_path", type=str, default="../base_imgs/workpls_v2.png")
    parser.add_argument("--image_path", type=str, default="../base_imgs/gum.png")
    parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")
    parser.add_argument("--gaze_start", type=str, default=f"[{717},{254}]")
    parser.add_argument("--gaze_end", type=str, default=f"[{424},{286}]")
    # parser.add_argument("--image_path", type=str, default="../base_imgs/zz.png")
    # parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")
    # parser.add_argument("--gaze_start", type=str, default=f"[{485},{329}]")
    # parser.add_argument("--gaze_end", type=str, default=f"[{189},{362}]")
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

    aa = time.time()
    preprocessed_image = preprocess(image)
    bb = time.time()
        
    print("encoder preprocess time:", bb - aa)

    for i in range(2):
        a = time.time()
        features = encoder(preprocessed_image)
        b = time.time()

        mask_input = torch.tensor(np.zeros((1, 1, 256, 256), dtype=np.float32)).cuda()
        has_mask_input = torch.tensor(np.zeros(1, dtype=np.float32)).cuda()

        point_coords = torch.randint(low=0, high=1024, size=(32, 1, 2), dtype=torch.float).cuda()
        point_labels = torch.randint(low=0, high=4, size=(32, 1), dtype=torch.float).cuda()

        c = time.time()
        decoder(features, point_coords, point_labels, mask_input, has_mask_input)
        d = time.time()

        print("prep encoder time:", b - a)
        print("prep decoder time:", d - c)


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
        
        gaze_pitchyaw, rvec, tvec = estimate_gaze_trt(frame, landmark, trt_gaze_estimation, timer)        
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
    

def prime_yolo(trt_yolo):
    image_path = "../base_imgs/psycho_out.png"
    raw_image = np.array(Image.open(image_path).convert("RGB"))
    image_yolo = cv2.resize(raw_image, (640, 640)) # must be (640, 640) to be compatible with engine
    expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
    yolo_img = torch.Tensor(expanded_img).cuda()
    trt_yolo(yolo_img)

def main():
    y = time.time()
    args = get_cli_args()

    # point processing
    args.gaze_start = yaml.safe_load(args.gaze_start)
    args.gaze_end = yaml.safe_load(args.gaze_end)

    # vit initialization
    trt_encoder_path = "engines/vit/encoder_fp32_k9.engine"
    trt_decoder_path = "engines/vit/decoder_fp32_k9.engine"
    efficientvit_sam = create_sam_model(args.model, True, None).cuda().eval()
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, trt_encoder_path=trt_encoder_path, trt_decoder_path=trt_decoder_path)

    # gaze initialization
    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    
    trt_face_detection = load_face_detection_engine("engines/gaze/face_detection_fp32_k9.engine")
    trt_landmark_detection = load_landmark_detection_engine("engines/gaze/landmark_detection_fp32_k9.engine")
    trt_gaze_estimation = load_gaze_estimation_engine("engines/gaze/gaze_estimation_fp32_k9.engine")


    timer = Timer()
    timer.start_record("whole_pipeline")
    CURRENT_TIMESTAMP = timer.get_current_timestamp()

    # yolo initialization
    trt_yolo = load_yolo_engine("engines/yolo/yolo_fp32_k9.engine")

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
    gaze_points, gaze_mask = get_pixels_on_line(raw_image, args.gaze_start, args.gaze_end)
    q = time.time()
    # run vit model
    c = time.time()
    masks = efficientvit_mask_generator.generate(raw_image, gaze_points)
    d = time.time()

    # run yolo model
    qq = time.time()
    image_yolo = cv2.resize(raw_image, (640, 640)) # must be (640, 640) to be compatible with engine
    expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
    r = time.time()
    yolo_img = torch.Tensor(expanded_img).cuda()
    ss = time.time()
    predictions = trt_yolo(yolo_img)
    s = time.time()
    bounding_boxes = visualize_bounding_boxes(raw_image, predictions, raw_image.shape[:2])
    t = time.time()
    u = time.time()

    timer.start_record("visualize")
    n = time.time()
    # show_frame = visualize_simple(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
    show_frame = visualize(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
    o = time.time()
    timer.end_record("visualize")
    timer.end_record("whole_pipeline")
    # show_frame = timer.print_on_image(show_frame)
    
    # visualize
    v = time.time()
    # show_anns(masks)
    raw_image = show_one_ann(masks, gaze_mask, bounding_boxes, args.gaze_start, raw_image)
    
    w = time.time()
    plt.axis("off")
    plt.imshow(raw_image)
    xx = time.time()
    plt.savefig(f"{args.output_path}", format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    x = time.time()

    print("full without load time:", time.time() - a)
    print(f"saving img to {args.output_path}")
    
    print()

    print("encoder/decoder priming run:", hhh - ggg)
    print("all gaze engines priming run:", fff - eee)
    print("yolo priming run:", zzz - yyy)

    print()

    print("load img:", b - a)
    print("resize img:", bb - b)
    print("generate masks:", d - c)
    print("detect face (primed):", f - e)
    print("smooth + extract face (primed):", h - g)
    print("detect landmark (primed):", j - i)
    print("smooth landmark (primed):", k - j)
    print("detect gaze (primed):", l - k)
    print("smooth gaze (primed):", m - l)
    print("visualize gaze:", o - n)
    print("create plots:", v - o)
    print("get gaze mask:", q - p)
    print("prep yolo img:", ss - qq)
    print("yolo pred:", s - ss)
    print("draw and get yolo boxes:", t - s)
    print("segment one mask:", w - v)

    print()

    print("display image:", xx - w)
    print(f"save to file ({args.output_path}):", x - w)
    print("non-load total:", w - e)
    print("load total:", z - y)

if __name__ == "__main__":
        main()

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