import numpy as np
import time

from typing import Any, Union, Tuple, Optional, List, Dict

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

# --- vit visualizers ---
def load_image(data_path: str, mode="rgb") -> np.ndarray:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return np.array(img)

def cat_images(image_list: List[np.ndarray], axis=1, pad=20) -> np.ndarray:
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

def show_anns_original(anns) -> None:
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    print("num sorted annotations:", len(sorted_anns))
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # color_mask = np.concatenate([(0, 0, 255), [0.35]])

        img[m] = color_mask
    ax.imshow(img)

def show_anns(anns) -> None:
    if len(anns) == 0:
        return
    print("num annotations:", len(anns))
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0]["segmentation"].shape[0], anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # color_mask = np.concatenate([(0, 0, 255), [0.35]])

        img[m] = color_mask
    ax.imshow(img)

def check_self(mask, eye_loc):
    rev = (eye_loc[1], eye_loc[0])
    return mask[rev] == 1

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

def show_one_ann_original(anns, line_mask, bbs, center_pix) -> None:
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

def show_one_ann_no_rles(anns, line_mask, bbs, center_pix) -> None:
    if len(anns) == 0:
        return
    
    print("num anns:", len(anns))

    img = np.ones((anns[0]["segmentation"].shape[0], anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    cv2.circle(img, (0,0), 5, (0, 255, 0), thickness=20)

    percentage_to_mask = {}
    for i, ann in enumerate(anns):
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
        mask = anns[mask_ix]['segmentation']
        img[mask] = color_mask
        print("point:", point)
        p = (point[1], point[0])
        plt.scatter(*p, color='blue', marker='*', s=500, edgecolors="white", linewidths=1)

    ax = plt.gca()
    ax.set_autoscale_on(False) 
    
    img[line_mask] = np.concatenate([[0, 255, 0], [0.5]])

    print("img.shape:", img.shape)

    ax.imshow(img) # this is important to keep the segmentations on the img

def show_one_ann(anns, line_mask, bbs, center_pix, raw_image) -> None:
    a = time.time()
    alpha_channel = np.ones((raw_image.shape[0], raw_image.shape[1], 1), dtype=np.uint8) * 255
    raw_image = np.concatenate((raw_image, alpha_channel), axis=2)
    b = time.time()

    print("create alpha channel:", b - a)

    if len(anns) == 0:
        return
    
    print("num anns:", len(anns))
    print("img.shape:", raw_image.shape)

    c = time.time()
    percentage_to_mask = {}
    for i, ann in enumerate(anns):
        m = ann["segmentation"]
        if check_self(m, center_pix): # dont want yourself
            continue
        
        percentage_intersection = intersects_bb(m, bbs)
        percentage_to_mask[percentage_intersection] = i # v low chance of same thing, in this case, j replace
                
    color_mask = np.concatenate([[255, 0, 0], [90]])
    d = time.time()
    print("create percentage overlap info:", d - c)

    if len(percentage_to_mask) == 0:
        print("EMPTY PERCENTAGE TO MASK???")
    else:
        max_percentage = max(percentage_to_mask)
        print("max percentage_overlap:", max_percentage, percentage_to_mask)
        mask_ix = percentage_to_mask[max_percentage]
        mask = anns[mask_ix]['segmentation']
        e = time.time()
        intersection = np.logical_and(mask, line_mask)
        ix = np.argwhere(intersection)
        f = time.time()
        print("find intersection point:", f - e)
        point = ix[0]
        g = time.time()
        raw_image[mask] = color_mask
        h = time.time()
        print("set mask:", h - g)
        cv2.drawMarker(raw_image, point[::-1], color=(255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=12, thickness=2) # star the segment point
        i = time.time()
        print("draw marker:", i - h)

    
    raw_image[line_mask] = np.concatenate([[0, 255, 0], [255]])

    return raw_image

def draw_binary_mask(raw_image: np.ndarray, binary_mask: np.ndarray, mask_color=(0, 0, 255)) -> np.ndarray:
    color_mask = np.zeros_like(raw_image, dtype=np.uint8)
    color_mask[binary_mask == 1] = mask_color
    mix = color_mask * 0.5 + raw_image * (1 - 0.5)
    binary_mask = np.expand_dims(binary_mask, axis=2)
    canvas = binary_mask * mix + (1 - binary_mask) * raw_image
    canvas = np.asarray(canvas, dtype=np.uint8)
    return canvas

def draw_bbox(
    image: np.ndarray,
    bbox: List[List[int]],
    color: str or List[str] = "g",
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
    points: List[List[int]],
    color: str or List[str] = "g",
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