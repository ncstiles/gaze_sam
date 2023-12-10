import numpy as np
import time

from typing import Any, Union, Tuple, Optional, List, Dict

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

MAUVE = np.array([177, 156, 217])
PURPLE = (217,156, 177)
TEAL = np.array([255, 244, 79])
YELLOW = np.array([79, 244, 255])
MUTED_BLUE = (159, 113, 59)
GREY = (128, 128, 128)

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

def overlay_old(raw_image, old_mask, old_line, old_point, old_box):

    if old_mask is not None: # only draw mask if its associated with a point, avoid bad mask quality
        cv2.drawMarker(raw_image, old_point, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=10) # star the segment point     

        color_mask = np.zeros_like(raw_image, dtype=np.uint8)
        color_mask[old_mask == 1] = YELLOW
        mix = color_mask * 0.5 + raw_image * (1 - 0.5)
        expanded_mask = np.expand_dims(old_mask, axis=2)
        canvas = expanded_mask * mix + (1 - expanded_mask) * raw_image
        canvas = np.asarray(canvas, dtype=np.uint8)
        raw_image = canvas

    if old_line:
        start, end = old_line[0], old_line[-1]
        cv2.line(raw_image, start, end, GREY, 2)
    
    if old_box is not None:
        x1, y1, x2, y2 = old_box
        cv2.rectangle(raw_image, (x1, y1), (x2, y2), MUTED_BLUE, 2)

    return raw_image

def show_one_box_ann(masks, gaze_start, gaze_end, bbs, raw_image) -> None:        
    print("bbs:", bbs)
    best_mask, bb_ix, largest_overlap = None, None, 0
    bb_areas = (bbs[:, 2] - bbs[:, 0] + 1) * (bbs[:, 3] - bbs[:, 1] + 1)
    print("whole masks shape", len(masks), len(bbs))
    if len(masks) != len(bbs):
        print("no match")
        return raw_image, None, None, None

    for i, bb in enumerate(bbs):
        mask = masks[i][0]
        if bb[0] <= gaze_start[0] <= bb[2] and bb[1] <= gaze_start[1] <= bb[3]: # bounding box contains person
            continue
        print("masks shape:", masks.shape, mask.shape)
        roi = mask[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
        mask_overlap = np.count_nonzero(roi)
        print("mask overlap:", mask_overlap)
        percentage_overlap = mask_overlap / bb_areas[i]
        if percentage_overlap > largest_overlap:
            best_mask = mask
            largest_overlap = percentage_overlap
            bb_ix = i
            print("mew largest overlap:", largest_overlap)
    
    focus_point = None
    mask = None
    bounding_box = None 
    if best_mask is not None: # default to the largest mask on the gaze line if non intersect with a box
        mask = best_mask
        # potential_focus_points = []
        
        # for i, point in enumerate(gaze_points):
        #     x1, y1 = point
        #     if mask[y1, x1]:
        #         potential_focus_points.append(point)
                
        # # get the one in the center
        # if potential_focus_points:
        #     focus_point = potential_focus_points[len(potential_focus_points)//2]
        #     cv2.drawMarker(raw_image, focus_point, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=10) # star the segment point

        color_mask = np.zeros_like(raw_image, dtype=np.uint8)
        color_mask[mask == 1] = YELLOW
        mix = color_mask * 0.5 + raw_image * (1 - 0.5)
        expanded_mask = np.expand_dims(mask, axis=2)
        canvas = expanded_mask * mix + (1 - expanded_mask) * raw_image
        canvas = np.asarray(canvas, dtype=np.uint8)
        raw_image = canvas

        print("bbs", bbs)
        print("bb ix:", bb_ix)
        bounding_box = bbs[bb_ix]
        x1, y1, x2, y2 = bounding_box
        print("best bb:", bounding_box)
        cv2.rectangle(raw_image, (x1, y1), (x2, y2), MUTED_BLUE, 2)


    cv2.line(raw_image, gaze_start, gaze_end, GREY, 2)
    print("line start:, line end:", gaze_start, gaze_end)

    return raw_image, mask, focus_point, bounding_box


def show_one_ann(anns, line_points, line_start, line_end, bbs, center_pix, raw_image) -> None:
    print()
    new_bbs = [] # don't add bounding box segmenting the gazing person
    for row in bbs:
        if not(row[0] <= center_pix[0] <= row[1] and row[2] <= center_pix[1] <= row[3]):
            new_bbs.append(row)
    
    bbs = np.array(new_bbs)
    print()
    print("~ extracting one mask ~")
    if len(anns) == 0:
        print("NO MASK SEGMENTATIONS FOUND")
        return raw_image, None, [], None, None
    
    if len(line_points) == 0:
        print("NO GAZE, skipping masking")
        return raw_image, None, [], None, None
    
    print("num anns:", len(anns))
    print("img.shape:", raw_image.shape)

    c = time.time()
    intersection_area = np.zeros((len(anns), len(bbs))) # keep track of num shared pixels between each (mask, bounding box) combo
    mask_area = np.zeros(len(anns)) # keep track of num pixels inside each mask

    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        if check_self(mask, center_pix): # never want to segment yourself, so set its val to be lowest
            intersection_area[i] = [-1] * len(bbs)
            continue
        else:
            mask_area[i] = np.count_nonzero(mask) # don't want to default to yourself again
        for j, bb in enumerate(bbs):
            roi = mask[bb[2]:bb[3]+1, bb[0]:bb[1]+1]
            intersection_area[i][j] = np.count_nonzero(roi)

    # (x2 - x1 + 1) * (y2 - y1 + 1) to get num pixels inside each bounding box
    best_mask_in_box = False # case when bbs is empty
    focus_point = None
    if len(bbs) > 0:
        bb_area = (bbs[:, 1] - bbs[:, 0] + 1) * (bbs[:, 3] - bbs[:, 2] + 1)

        percentage_intersection_area = intersection_area / bb_area
        max_ix = np.argmax(percentage_intersection_area)

        r, c = max_ix // len(bbs), max_ix % len(bbs)
        print("max vals:", "r:", r, "c:", c, percentage_intersection_area[r][c])

        best_mask_in_box = percentage_intersection_area[r][c] > 0.60 # max intersection between mask and box is 0

    if best_mask_in_box: # default to the largest mask on the gaze line if non intersect with a box
        mask = anns[r]['segmentation']
        potential_focus_points = []
        
        for i, point in enumerate(line_points):
            x1, y1 = point
            if mask[y1, x1]:
                potential_focus_points.append(point)
                
        # get the one in the center
        if potential_focus_points:
            focus_point = potential_focus_points[len(potential_focus_points)//2]
            cv2.drawMarker(raw_image, focus_point, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=10) # star the segment point

            color_mask = np.zeros_like(raw_image, dtype=np.uint8)
            color_mask[mask == 1] = PURPLE
            mix = color_mask * 0.5 + raw_image * (1 - 0.5)
            expanded_mask = np.expand_dims(mask, axis=2)
            canvas = expanded_mask * mix + (1 - expanded_mask) * raw_image
            canvas = np.asarray(canvas, dtype=np.uint8)
            raw_image = canvas

    cv2.line(raw_image, line_start, line_end, (0, 0, 255), 2)
    print("line start:, line end:", line_start, line_end)

    bounding_box = None
    return raw_image, mask, [line_start, line_end], focus_point, bounding_box

def show_anns(raw_image, anns) -> None:
    if len(anns) == 0:
        return raw_image
    print("num annotations:", len(anns))
    
    alpha_channel = np.ones((raw_image.shape[0], raw_image.shape[1], 1), dtype=np.uint8) * 255
    raw_image = np.concatenate((raw_image, alpha_channel), axis=2)
    for ann in anns:
        m = ann["segmentation"]
        raw_image[m] = np.concatenate([np.random.random(3) * 255, [0.35 * 255]])
    
    return raw_image

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

def show_one_ann_numpyless(anns, line_mask, bbs, center_pix, raw_image) -> None:
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