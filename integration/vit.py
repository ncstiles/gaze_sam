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

from vit_utils import load_image, cat_images, show_anns, draw_bbox, draw_scatter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="l1")
    # parser.add_argument("--image_path", type=str, default="../base_imgs/fig/cat.jpg")
    parser.add_argument("--image_path", type=str, default="../base_imgs/workpls.png")
    parser.add_argument("--output_path", type=str, default=f"out/{time.time()}.png")

    args, _ = parser.parse_known_args()

    # build model
    efficientvit_sam = create_sam_model(args.model, True, None).cuda().eval()

    trt_encoder_path = "engines/vit/encoder.engine"
    trt_decoder_path = "engines/vit/decoder.engine"
    
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, trt_encoder_path=trt_encoder_path, trt_decoder_path=trt_decoder_path)

    # load image
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    if raw_image.shape[0] * raw_image.shape[1] > 1280 * 720:
        raw_image = cv2.resize(raw_image, (1280, 720))

    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    start = time.time()
    masks = efficientvit_mask_generator.generate(raw_image)
    end = time.time()
    
    print("mask generation time:", end - start)
    plt.figure(figsize=(20, 20))
    plt.imshow(raw_image)
    show_anns(masks)
    plt.axis("off")

    print(f"saving img to {args.output_path}")
    plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    main()