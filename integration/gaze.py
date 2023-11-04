
import argparse
import math

import cv2
import numpy as np

import time
from torch2trt import TRTModule
import tensorrt as trt

import torch

import sys
sys.path.append("../")
from proxylessnas.proxyless_gaze.deployment.onnx.demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from proxylessnas.proxyless_gaze.deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter

from gaze_utils import *

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument("--source", default="/dev/video0", type=str)
    parser.add_argument("--save-video", default=None, type=str, required=False)
    parser.add_argument("--image-path", default="../base_imgs/workpls.png")
    parser.add_argument("--output-path", default=f"out/{time.time()}.png")
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    
    trt_face_detection = load_face_detection_engine("engines/gaze/face_detection.engine")
    trt_landmark_detection = load_landmark_detection_engine("engines/gaze/landmark_detection.engine")
    trt_gaze_estimation = load_gaze_estimation_engine("engines/gaze/gaze_estimation.engine")
    
    # cap = cv2.VideoCapture(0)
    timer = Timer()

    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    
    # if args.save_video is not None:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     writer = cv2.VideoWriter(args.save_video, fourcc, 20.0, (640, 480))
    
    cnt = 0

    raw_image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
    frame = np.array(raw_image)

    # while True:
    # ret, frame = cap.read()
    # if not ret or frame is None:
    #     break
    timer.start_record("whole_pipeline")
    show_frame = frame.copy()
    CURRENT_TIMESTAMP = timer.get_current_timestamp()
    # cnt += 1
    # if cnt % 2 == 1:
    faces = detect_face_trt(frame, trt_face_detection, timer)
    if faces is not None:
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
        face = np.array([x1,y1,x2,y2,face[-1]])
        landmark, landmark_on_cropped, cropped = detect_landmark_trt(frame, face, trt_landmark_detection, timer)
        landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
        
        # gaze_pitchyaw, rvec, tvec = estimate_gaze(frame, landmark, gaze_estimation_session)
        # print("final gaze metrics:", gaze_pitchyaw.shape, rvec.shape, tvec.shape)
        gaze_pitchyaw, rvec, tvec = estimate_gaze_trt(frame, landmark, trt_gaze_estimation, timer)
        
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
        timer.start_record("visualize")
        show_frame = visualize(show_frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
        timer.end_record("visualize")
    timer.end_record("whole_pipeline")
    show_frame = timer.print_on_image(show_frame)
    # if args.save_video is not None:
    #     writer.write(show_frame)
    # cv2.imshow("onnx_demo", show_frame)
    cv2.imwrite(args.output_path, show_frame)
    # code = cv2.waitKey(1)
    # if code == 27:
    #     break
