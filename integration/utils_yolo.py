import cv2
import numpy as np
import time

def visualize_bounding_boxes_numpy(img, predictions, orig_shape):
    h, w = orig_shape
    YOLO_LEN = 640  # YOLO model takes in (640, 640) image, so boxes are proportional to this SF
    W_SF, H_SF = w / YOLO_LEN, h / YOLO_LEN

    _, pred_boxes, _, _ = predictions
    pred_boxes = pred_boxes.detach().cpu().numpy()[0]

    # Scale bounding boxes using vectorized NumPy operations
    x1, y1, x2, y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x1 = (x1 * W_SF).astype(int)
    x2 = (x2 * W_SF).astype(int)
    y1 = (y1 * H_SF).astype(int)
    y2 = (y2 * H_SF).astype(int)

    scaled_boxes = np.column_stack((x1, y1, x2, y2))

    for (x1, y1, x2, y2) in scaled_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return scaled_boxes

def visualize_bounding_boxes(img, predictions, orig_shape):
    h, w = orig_shape
    _, pred_boxes, _, _ = predictions
    pred_boxes = pred_boxes.detach().cpu().numpy()[0]

    scaled_boxes = []
    for bounding_box in pred_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bounding_box]
        x1 = int(x1 * w/640)
        x2 = int(x2 * w/640)
        y1 = int(y1 * h/640)
        y2 = int(y2 * h/640)
        if x1 == x2 == y1 == y2: # optimization: don't process 0-pixel bounding boxes
            continue
        scaled_boxes.append([x1, x2, y1, y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    print("number of bounding boxes:", len(scaled_boxes))
    return np.array(scaled_boxes)

def get_bounding_boxes(predictions, orig_shape):
    h, w = orig_shape
    YOLO_LEN = 640 # yolo model takes in (640, 640) image, so boxes are proportional to this SF
    W_SF, H_SF = w/YOLO_LEN, h/YOLO_LEN

    _, pred_boxes, _, _ = predictions
    pred_boxes = pred_boxes.detach().cpu().numpy()[0]
    
    scaled_boxes = []
    for bounding_box in pred_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bounding_box]
        x1 = int(x1 * W_SF)
        x2 = int(x2 * W_SF)
        y1 = int(y1 * H_SF)
        y2 = int(y2 * H_SF)
        scaled_boxes.append((x1, y1, x2, y2))

    return scaled_boxes
