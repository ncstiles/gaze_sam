import cv2

def visualize_bounding_boxes(img, predictions, orig_shape):
    h, w = orig_shape
    _, pred_boxes, _, _ = predictions
    pred_boxes = pred_boxes.detach().cpu().numpy()[0]
        
    for bounding_box in pred_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in bounding_box]
        x1 = int(x1 * w/640)
        x2 = int(x2 * w/640)
        y1 = int(y1 * h/640)
        y2 = int(y2 * h/640)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
