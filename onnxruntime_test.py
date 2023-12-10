import time
import cv2
import numpy as np
import onnxruntime 

from super_gradients.common.object_names import Models
from super_gradients.training import models

from efficient_vit.demo_sam_model import load_image


filename = 'yolo_onnxruntime.onnx'
# model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
# export_result = model.export(filename)

# print(export_result)
# print(export_result.input_image_shape[1])
# print(export_result.input_image_shape[0])

img = load_image('base_imgs/gum.png')

# image = cv2.resize(img, (export_result.input_image_shape[1], export_result.input_image_shape[0]))
image = cv2.resize(img, (640, 640))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession(filename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
print("inputs:", inputs)
print("outputs:", outputs)
start = time.time()
result = session.run(outputs, {inputs[0]: image_bchw})
end = time.time()
print("time to do inference on one image:", end - start)

print(result[0].shape)
print(result[1].shape)
print(result[2].shape)
print(result[3].shape)

num_predictions, pred_boxes, pred_scores, pred_classes = result

assert num_predictions.shape[0] == 1, "Only batch size of 1 is supported by this function"

num_predictions = int(num_predictions.item())
pred_boxes = pred_boxes[0, :num_predictions]
pred_scores = pred_scores[0, :num_predictions]
pred_classes = pred_classes[0, :num_predictions]

img2 = cv2.resize(image, (1280, 720))

x_sf, y_sf = 1280/ 640, 720/640 

for (x1, y1, x2, y2, class_score, class_index) in zip(pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3], pred_scores, pred_classes):
    # image = DetectionVisualization.draw_box_title(
    #     image_np=image,
    #     x1=int(x1),
    #     y1=int(y1),
    #     x2=int(x2),
    #     y2=int(y2),
    #     class_id=class_index,
    #     class_names=class_names,
    #     color_mapping=color_mapping,
    #     box_thickness=2,
    #     pred_conf=class_score,
    # )
    cv2.rectangle(img2, (int(x1 * x_sf),int(y1 * y_sf)), (int(x2 * x_sf),int(y2 * y_sf)), (0, 255, 0), 2)



cv2.imwrite("tester_onnxout.png", img2)

# result[0].shape, result[1].shape, result[2].shape, result[3].shape