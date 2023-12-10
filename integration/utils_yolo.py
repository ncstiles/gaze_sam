import cv2
import numpy as np
import time

def visualize_bounding_boxes(img, predictions, orig_shape):
    h, w = orig_shape
    print("orig shape:", h, w)
    print("img shape:", img.shape)
    YOLO_LEN = 640.0  # YOLO model takes in (640, 640) image, so boxes are proportional to this SF
    W_SF, H_SF = w / YOLO_LEN, h / YOLO_LEN

    _, pred_boxes, _, _ = predictions
    pred_boxes = pred_boxes.detach().cpu().numpy()[0]

    # Scale bounding boxes using vectorized NumPy operations
    x1, y1, x2, y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    boxes = np.column_stack((x1, y1, x2, y2))

    # remove bounding boxes where x-coords = y-coords (ex: 0, 0, 0, 0)
    mask = ~np.all(boxes[:, 1:] == boxes[:, :-1], axis=1)
    nondup_boxes = boxes[mask] 

    scales = np.array([W_SF, H_SF, W_SF, H_SF])
    scaled_boxes = (nondup_boxes * scales).astype(int)

    # for x1,y1,x2,y2 in scaled_boxes:
    #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 3)

    # for box in scaled_boxes:
    #     print("boxxxxx:", box)

    # cv2.imwrite("out/nicole_test.png", img)
# 
    return scaled_boxes