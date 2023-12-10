from trt_sam import EfficientViTSamAutomaticMaskGenerator

from efficient_vit.efficientvit.sam_model_zoo import create_sam_model

from proxylessnas.proxyless_gaze.deployment.onnx.demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from proxylessnas.proxyless_gaze.deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter
from load_engine import *
from utils_gaze import *
from utils_vit import *
from utils_yolo import *

from shapely.geometry import LineString, box

import cv2

trt_encoder_path = "engines/vit/encoder_fp32.engine"
trt_decoder_path = "engines/vit/box_decoder.engine"
efficientvit_sam = create_sam_model("l0", True, None).cuda().eval()
efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam, trt_encoder_path=trt_encoder_path, trt_decoder_path=trt_decoder_path)
timer = Timer()

gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
trt_face_detection = load_face_detection_engine("engines/gaze/face_detection_fp16.engine") 
trt_landmark_detection = load_landmark_detection_engine("engines/gaze/landmark_detection_fp16.engine")
trt_gaze_estimation = load_gaze_estimation_engine("engines/gaze/gaze_estimation_fp16.engine")

trt_yolo = load_yolo_engine("engines/yolo/yolo_fp32.engine") 


# Create a video capture object
cap = cv2.VideoCapture("final.mp4")
# Check if the video is opened successfully
if not cap.isOpened():
    print('Error opening video file')

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object with H.264 codec using a different FourCC code
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # or 'AVC1'

print("fps:", fps)
ret, frame = cap.read()
height, width, _ = frame.shape
out = cv2.VideoWriter('out_thurs_final.mp4', fourcc, fps, (width, height)) ##############################


def get_bbs_intersecting_gaze(gaze_start, gaze_end, box_coords):
    # if line_coords.shape[0] == 1: #doesn't work on 1 point
    #     on_line = [False] * box_coords.shape[0]
    # else:
    on_line = []
    line = LineString((gaze_start, gaze_end))

    for single_box in box_coords:
        b = box(single_box[0], single_box[1], single_box[2], single_box[3])
        on_line.append(line.intersects(b))

    return box_coords[on_line]

aaaaaaa = time.time()
count = 0
old_point, old_mask, old_box = None, None, None
while True:
    a  = time.time()
    count += 1
    CURRENT_TIMESTAMP = timer.get_current_timestamp()
    ret, frame = cap.read()

    raw_image = frame
    masks = []
    print("...")
    if not ret or frame is None:
        break

    faces = detect_face_trt(frame, trt_face_detection, timer)
    if faces is None:
        print("NO FACES")


    f = time.time()
    # sometimes no face, want these vars to still exist
    g, h, i, j, k, l, m = 0, 0, 0, 0, 0, 0, 0

    # run yolo model
    qq = time.time()
    image_yolo = cv2.resize(frame, (640, 640)) # must be (640, 640) to be compatible with engine
    expanded_img = np.transpose(np.expand_dims(image_yolo, axis=0), (0, 3, 1, 2))
    yolo_img = torch.Tensor(expanded_img).cuda()
    ss = time.time()
    predictions = trt_yolo(yolo_img)
    s = time.time()
    bounding_boxes = visualize_bounding_boxes(frame, predictions, frame.shape[:2])
    t = time.time()

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

        frame, gaze_start, gaze_end = visualize_simple(frame, face, landmark, gaze_pitchyaw, [rvec, tvec])

        gaze_start, gaze_end = get_pixels_on_line(frame, gaze_start, gaze_end)

        filtered_boxes = get_bbs_intersecting_gaze(gaze_start, gaze_end, bounding_boxes)        

        q = time.time()
        # run vit model
        c = time.time()
        if count % 2 == 1:
            masks = efficientvit_mask_generator.generate(frame, [], filtered_boxes)
            d = time.time()
            print("time for all but process:", d -a)
            frame, old_mask, old_point, old_box = show_one_box_ann(masks, gaze_start, gaze_end, filtered_boxes, raw_image)
        else:
            frame = overlay_old(frame, old_mask, (gaze_start, gaze_end), old_point, old_box)

        out_img = cv2.resize(frame, (width, height))
        b = time.time()
        print('time for frame:', b - a)
        out.write(out_img)
    else:
        out.write(raw_image)

bbbbbbb = time.time()

print("total frames evaluated:", count)
print(f"\n\ntotal time: {bbbbbbb - aaaaaaa}")
print("frames per second:", count / (bbbbbbb - aaaaaaa))
# Release the VideoWriter object
out.release()

# Release the VideoCapture object
cap.release()

# Destroy any OpenCV windows if needed
cv2.destroyAllWindows()
