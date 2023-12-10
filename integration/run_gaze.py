import argparse
import math

import cv2
import numpy as np
import onnxruntime

import os
import numpy as np
import platform
import time
import ctypes
import cv2

# from demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
import sys
sys.path.append("../")
from proxylessnas.proxyless_gaze.deployment.onnx.smoother import GazeSmoother, LandmarkSmoother, OneEuroFilter

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

class Timer():
    def __init__(self, saved_n=100):
        self.record = {}
        self.start_time = {}
        self.end_time = {}
        self.current_name = None
        self.saved_n = saved_n
        # self.weights = list(range(saved_n))
        # self.weights.reverse()
        # self.weights = np.exp(-np.array(self.weights))
        if "Windows" in platform.platform():
            freq = ctypes.c_longlong(0)
            ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
            self.freq = freq.value
            self.get_current_timestamp = self.get_current_timestamp_windows
        else:
            self.get_current_timestamp = self.get_current_timestamp_linux
    
    def get_current_timestamp_windows(self):
        freq = ctypes.c_longlong(0)
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(freq))
        return freq.value / self.freq

    def get_current_timestamp_linux(self):
        return time.time()
    
    def start_record(self, name=None):
        if name is None:
            name = "default"
        self.current_name = name
        self.start_time[name] = self.get_current_timestamp()
        
    def end_record(self, name=None):
        if name is None:
            name = self.current_name
        self.end_time[name] = self.get_current_timestamp()
        if name not in self.record:
            # self.record[name] = [0] * self.saved_n
            self.record[name] = []
        self.record[name].append(self.end_time[name] - self.start_time[name])
        if len(self.record[name]) > self.saved_n:
            self.record[name].pop(0)

    def get_record_s(self, name=None) -> float:
        if name is None:
            name = self.record.keys()
        return sum([np.mean(self.record[n]) for n in name])
    
    def get_record_ms(self, name=None) -> float:
        return round(self.get_record_s(name) * 1000.0, 2)
    
    def clear_all(self):
        self.start_time = {}
        self.end_time = {}
        self.record = {}
    
    def print_on_image(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0,0,0)
        cv2.putText(img, f"Size: {img.shape[1]}x{img.shape[0]}", (10,35), font, 0.8, color, 2)
        cv2.putText(img, f"FPS: {int(1/(self.get_record_s(['whole_pipeline'])+1e-5))}", (10,60), font, 0.8, color, 2)
        cv2.putText(img, f"Total: {self.get_record_ms(['whole_pipeline'])}ms", (10,85), font, 0.8, color, 2)
        
        cnt = 0
        if "face_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection: {self.get_record_ms(['face_detection'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "face_detection_postprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection NMS: {self.get_record_ms(['face_detection_postprocess'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        
        if "landmark_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Landmark: {self.get_record_ms(['landmark_detection'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze: {self.get_record_ms(['gaze_estimation'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation_preprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze Preprocess: {self.get_record_ms(['gaze_estimation_preprocess'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "visualize" in self.record:
            cnt += 1
            cv2.putText(img, f"Visualize: {self.get_record_ms(['visualize'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        
        return img
      
def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)


face_model = np.float32([
    [-63.833572,  63.223045,  41.1674  ], # RIGHT_EYEBROW_RIGHT,
    [-12.44103 ,  66.60398 ,  64.561584], # RIGHT_EYEBROW_LEFT,
    [ 12.44103 ,  66.60398 ,  64.561584], # LEFT_EYEBROW_RIGHT, 
    [ 63.833572,  63.223045,  41.1674  ], # LEFT_EYEBROW_LEFT, 
    [-49.670784,  51.29701 ,  37.291245], # RIGHT_EYE_RIGHT,
    [-16.738844,  50.439426,  41.27281 ], # RIGHT_EYE_LEFT,
    [ 16.738844,  50.439426,  41.27281 ], # LEFT_EYE_RIGHT,
    [ 49.670784,  51.29701 ,  37.291245], # LEFT_EYE_LEFT,
    [-18.755981,  13.184412,  57.659172], # NOSE_RIGHT,
    [ 18.755981,  13.184412,  57.659172], # NOSE_LEFT,
    [-25.941687, -19.458733,  47.212223], # MOUTH_RIGHT,
    [ 25.941687, -19.458733,  47.212223], # MOUTH_LEFT,
    [  0.      , -29.143637,  57.023403], # LOWER_LIP,
    [  0.      , -69.34913 ,  38.065376]  # CHIN
])

cam_w, cam_h = 640, 480
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / np.tan(60 / 2 * np.pi / 180)
f_y = f_x
camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

def yolox_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def extract_critical_landmarks(landmark, pt_num=14):
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    critical_landmarks = landmark[TRACKED_POINTS]
    critical_landmarks = np.array(critical_landmarks)
    return critical_landmarks

def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec

def vec_to_euler(x,y,z):
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return theta, phi

def rtvec_to_euler(rvec, tvec, unit="radian"):
    rvec_matrix = cv2.Rodrigues(rvec)[0]
    proj_matrix = np.hstack((rvec_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    if unit == "degree":
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
    return pitch, yaw, roll

def estimateHeadPose(landmarks, iterate=False):
    landmarks = extract_critical_landmarks(landmarks)
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera_matrix, camera_distortion)
    ## further optimize
    # if iterate:
    #     ret, rvec, tvec = cv2.solvePnP(facePts, landmarks, camera_matrix, camera_distortion, rvec, tvec, True)
    return rvec, tvec

def normalizeDataForInference(img, hr, ht):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm_eye = 700 # normalized distance between eye and camera
    distance_norm_face = 1200 # normalized distance between face and camera
    roiSize_eye = (60, 60) # size of cropped eye image
    roiSize_face = (120, 120) # size of cropped face image
    # img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_u = img

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht # 3D positions of facial landmarks

    re = 0.5*(Fc[:,4] + Fc[:,5]).T # center of eye on the left side of image (right eye)
    le = 0.5*(Fc[:,6] + Fc[:,7]).T # center of eye on the right side of image (left eye)
    fe = (1./6.)*(Fc[:,4] + Fc[:,5] + Fc[:,6] + Fc[:,7] + Fc[:,10] + Fc[:,11]).T

    ## normalize each eye
    data = []
    for distance_norm, roiSize, et in zip([distance_norm_eye, distance_norm_eye, distance_norm_face], [roiSize_eye, roiSize_eye, roiSize_face], [re, le, fe]):
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T # rotation matrix R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera_matrix))) # transformation matrix
        
        img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
        # img_warped = cv2.equalizeHist(img_warped)
        data.append(img_warped)

        if distance_norm == distance_norm_face:
             data.append(R)
    return data

def detect_face(img, session, score_thr=0.5, input_shape=(160, 128)) -> np.ndarray:
    img, ratio = yolox_preprocess(img, input_shape)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    timer.start_record("face_detection")
    output = session.run(None, ort_inputs)
    timer.end_record("face_detection")
    timer.start_record("face_detection_postprocess")
    predictions = demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr)
    timer.end_record("face_detection_postprocess")
    if dets is not None:
        final_boxes, final_scores = dets[:, :4], dets[:, 4]
        return np.array([[*final_box, final_score] for final_box, final_score in zip(final_boxes, final_scores)])
    else:
        return None

def detect_landmark(img, face, session) -> np.ndarray:
    height, width = img.shape[:2]
    x1, y1, x2, y2 = map(int, face[:4])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2

    size = int(max([w, h]) * 1.11)
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    edx1 = max(0, -x1)
    edy1 = max(0, -y1)
    edx2 = max(0, x2 - width)
    edy2 = max(0, y2 - height)

    cropped = img[y1:y2, x1:x2]
    if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
        cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                        cv2.BORDER_CONSTANT, 0)

    input = cv2.resize(cropped, (112, 112))
    input = input.transpose((2,0,1)) / 255.0
    input = np.ascontiguousarray(input, dtype=np.float32)
    
    ort_inputs = {session.get_inputs()[0].name: input[None,:,:,:]}
    timer.start_record("landmark_detection")
    _, landmark = session.run(None, ort_inputs)
    timer.end_record("landmark_detection")
    pre_landmark = landmark[0].reshape(-1, 2) * [size, size]
    landmark_on_cropped = pre_landmark.copy()
    pre_landmark -= [edx1, edy1]
    pre_landmark[:, 0] += x1
    pre_landmark[:, 1] += y1
    return pre_landmark, landmark_on_cropped, cropped

def estimate_gaze(img, landmark, session) -> np.ndarray:
    timer.start_record("gaze_estimation_preprocess")
    rvec, tvec = estimateHeadPose(landmark)
    data = normalizeDataForInference(img, rvec, tvec)
    timer.end_record("gaze_estimation_preprocess")
    leye_image, reye_image, face_image, R = data

    leye_image = np.ascontiguousarray(leye_image, dtype=np.float32) / 255.0
    reye_image = np.ascontiguousarray(reye_image, dtype=np.float32) / 255.0
    face_image = np.ascontiguousarray(face_image, dtype=np.float32) / 255.0
    leye_image = np.transpose(np.expand_dims(leye_image, 0), (0,3,1,2))
    reye_image = np.transpose(np.expand_dims(reye_image, 0), (0,3,1,2))
    face_image = np.transpose(np.expand_dims(face_image, 0), (0,3,1,2))

    ort_inputs = {session.get_inputs()[0].name: leye_image,
                  session.get_inputs()[1].name: reye_image,
                  session.get_inputs()[2].name: face_image}
    timer.start_record("gaze_estimation")
    pred_pitchyaw_aligned = session.run(None, ort_inputs)[0][0]
    timer.end_record("gaze_estimation")
    pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
    pred_vec_aligned = euler_to_vec(*pred_pitchyaw_aligned)
    pred_vec_cam = np.dot(np.linalg.inv(R), pred_vec_aligned)
    pred_vec_cam /= np.linalg.norm(pred_vec_cam)
    pred_pitchyaw_cam = np.array(vec_to_euler(*pred_vec_cam))
    return pred_pitchyaw_cam, rvec, tvec

def visualize(img, face=None, landmark=None, gaze_pitchyaw=None, headpose=None):
    if face is not None:
        bbox = face[:4].astype(int)
        score = face[4]
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (0,255,0), 2)
        text = f'conf: {score * 100:.1f}%'
        txt_color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (bbox[0], bbox[1]-5), font, 0.5, txt_color, thickness=1)
    if landmark is not None:
        for i, (x, y) in enumerate(landmark.astype(np.int32)):
            cv2.circle(img, (x, y), 2, (255, 0, 0), thickness=-1)
    if gaze_pitchyaw is not None:
        eye_pos = landmark[-2:].mean(0)
        draw_gaze(img, eye_pos, gaze_pitchyaw, 300, 4)
    if headpose is not None:
        rvec = headpose[0]
        tvec = headpose[1]
        axis = np.float32([[50, 0, 0],
                            [0, 50, 0],
                            [0, 0, 50],
                            [0, 0, 0]])
    
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
        modelpts, _ = cv2.projectPoints(face_model, rvec, tvec, camera_matrix, camera_distortion)
        imgpts = imgpts.astype(int)
        modelpts = modelpts.astype(int)
        delta = modelpts[-1].ravel() - imgpts[-1].ravel()
        imgpts += delta
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[0].ravel()), (255, 0, 0), 3) # Blue x-axis
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 3) # Green y-axis
        cv2.line(img, tuple(imgpts[-1].ravel()), tuple(imgpts[2].ravel()), (0, 0, 255), 3) # Red z-axis
        # for i, pt in enumerate(modelpts):
        #     cv2.circle(img, tuple(pt.ravel()), 4, (0, 0, 255), thickness=-1)
        #     cv2.putText(img, f"{i}", tuple(pt.ravel()), font, 0.5, txt_color, thickness=1)
    return img

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument("--source", default="/dev/video0", type=str)
    parser.add_argument("--save-video", default=None, type=str, required=False)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    
    face_detection_session = onnxruntime.InferenceSession("/home/nicole/gaze_sam/integration/engine_creation/gaze/face_detection.onnx")
    landmark_detection_session = onnxruntime.InferenceSession("/home/nicole/gaze_sam/integration/engine_creation/gaze/landmark_detection.onnx")
    gaze_estimation_session = onnxruntime.InferenceSession("/home/nicole/gaze_sam/integration/engine_creation/gaze/gaze_estimation.onnx")
    
    # cap = cv2.VideoCapture(0)
    timer = Timer()

    gaze_smoother = GazeSmoother(OneEuroFilter, min_cutoff=0.01, beta=0.8)
    landmark_smoother = LandmarkSmoother(OneEuroFilter, pt_num=98, min_cutoff=0.1, beta=1.0)
    bbox_smoother = LandmarkSmoother(OneEuroFilter, pt_num=2, min_cutoff=0.0, beta=1.0)
    

    frame = cv2.imread("/home/nicole/gaze_sam/base_imgs/wall.png")

    timer.start_record("whole_pipeline")
    show_frame = frame.copy()
    CURRENT_TIMESTAMP = timer.get_current_timestamp()

    faces = detect_face(frame, face_detection_session)
    
    if faces is not None:
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        [[x1,y1],[x2,y2]] = bbox_smoother([[x1,y1],[x2,y2]], t=CURRENT_TIMESTAMP)
        face = np.array([x1,y1,x2,y2,face[-1]])
        landmark, landmark_on_cropped, cropped = detect_landmark(frame, face, landmark_detection_session)
        landmark = landmark_smoother(landmark, t=CURRENT_TIMESTAMP)
        gaze_pitchyaw, rvec, tvec = estimate_gaze(frame, landmark, gaze_estimation_session)
        gaze_pitchyaw = gaze_smoother(gaze_pitchyaw, t=CURRENT_TIMESTAMP)
        timer.start_record("visualize")
        show_frame = visualize(show_frame, face, landmark, gaze_pitchyaw, [rvec, tvec])
        timer.end_record("visualize")
    
    timer.end_record("whole_pipeline")
    show_frame = timer.print_on_image(show_frame)
    cv2.imwrite(f"a_del/onnx_demo_{time.time()}.png", show_frame)
