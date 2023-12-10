# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
import time

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from efficient_vit.efficientvit.apps.utils import parse_unknown_args
from efficient_vit.efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficient_vit.efficientvit.models.utils import build_kwargs_from_config
from efficient_vit.efficientvit.sam_model_zoo import create_sam_model


def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)


def cat_images(image_list: list[np.ndarray], axis=1, pad=20) -> np.ndarray:
    shape_list = [image.shape for image in image_list]
    max_h = max([shape[0] for shape in shape_list]) + pad * 2
    max_w = max([shape[1] for shape in shape_list]) + pad * 2

    for i, image in enumerate(image_list):
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        h, w, _ = image.shape
        crop_y = (max_h - h) // 2
        crop_x = (max_w - w) // 2
        canvas[crop_y : crop_y + h, crop_x : crop_x + w] = image
        image_list[i] = canvas

    image = np.concatenate(image_list, axis=axis)
    return image


def show_anns(anns) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    mask_color = random.sample([(0, 0, 255), (255, 0, 0), (0, 255, 0)], 1)
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas


def draw_bbox(
    image: np.ndarray,
    bbox: list[list[int]],
    color: str or list[str] = "g",
    linewidth=1,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in bbox]
    for (x0, y0, x1, y1), c in zip(bbox, color):
        plt.gca().add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=linewidth, edgecolor=c, facecolor=(0, 0, 0, 0)))
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image


def draw_scatter(
    image: np.ndarray,
    points: list[list[int]],
    color: str or list[str] = "g",
    marker="*",
    s=10,
    ew=0.25,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    if isinstance(color, str):
        color = [color for _ in points]
    for (x, y), c in zip(points, color):
        plt.scatter(x, y, color=c, marker=marker, s=s, edgecolors="white", linewidths=ew)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image

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

    tups = []

    print("len x vals before:", len(x_vals))
    gap = len(x_vals) // 20
    for i in range(0, len(x_vals), gap):
        x, y = x_vals[i], y_vals[i]
        cv2.circle(img, (x,y), 1, (255, 0, 0), thickness=1)
        tups.append((x,y))

    mask = np.zeros((H, W))
    mask[y_vals, x_vals] = 1

    return mask.astype(bool), tups

def main():
    print("inside main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--image_path", type=str, default="efficient_vit/assets/fig/cat.jpg")
    parser.add_argument("--output_path", type=str, default="segment_line_efficientvit_sam_demo.png")

    parser.add_argument("--mode", type=str, default="all", choices=["point", "box", "all"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--box", type=str, default=None)

    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # build model
    efficientvit_sam = create_sam_model(args.model, True, args.weight_url).cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
        efficientvit_sam, **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator)
    )

    # load image
    raw_image = np.array(Image.open(args.image_path).convert("RGB"))
    raw_image = cv2.resize(raw_image, (1280, 720))
    H, W, _ = raw_image.shape
    print(f"Image Size: W={W}, H={H}")

    tmp_file = f".tmp_{time.time()}.png"
    if args.mode == "all":
        masks = efficientvit_mask_generator.generate(raw_image)
        plt.figure(figsize=(20, 20))
        plt.imshow(raw_image)
        show_anns(masks)
        plt.axis("off")
        plt.savefig(args.output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.0)
    elif args.mode == "point":
        # start, end = (617, 288), (808, 242)
        start, end = (595, 361), (757, 396)
        _, points_on_line = get_pixels_on_line(raw_image, start, end)
        point_coords, point_labels = [], []

        print("inside point mode")
        print("H, W", H, W)

        point_coords = points_on_line
        point_labels = [1] * len(points_on_line)

        efficientvit_sam_predictor.set_image(raw_image)

        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=np.array(point_coords),
            point_labels=np.array(point_labels),
            multimask_output=args.multimask,
        )
        plots = [
            draw_scatter(
                draw_binary_mask(raw_image, binary_mask, np.random.random(3)),
                point_coords,
                color=["g" if l == 1 else "r" for l in point_labels],
                s=10,
                ew=0.25,
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    elif args.mode == "box":
        args.box = yaml.safe_load(args.box)
        efficientvit_sam_predictor.set_image(raw_image)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(args.box),
            multimask_output=args.multimask,
        )
        plots = [
            draw_bbox(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                [args.box],
                color="g",
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        Image.fromarray(plots).save(args.output_path)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()