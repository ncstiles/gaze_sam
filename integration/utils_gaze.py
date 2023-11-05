import argparse
import math

import cv2
import numpy as np

import time
import torch

import sys
sys.path.append("../")
from proxylessnas.proxyless_gaze.deployment.onnx.demo_utils import multiclass_nms, demo_postprocess, Timer, draw_gaze
from load_engine import load_face_detection_engine, load_gaze_estimation_engine, load_landmark_detection_engine

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

def visualize_simple(img, face=None, landmark=None, gaze_pitchyaw=None, headpose=None):
    if gaze_pitchyaw is not None:
        eye_pos = landmark[-2:].mean(0)
        draw_gaze(img, eye_pos, gaze_pitchyaw, 300, 4)
    
    return img

def detect_face_trt(img, face_detection_engine, timer, score_thr=0.5, input_shape=(160, 128)) -> np.ndarray:
    img, ratio = yolox_preprocess(img, input_shape)
    full_image = torch.Tensor(img[None, :, :, :]).cuda()
    # print("input shape:", full_image.shape)
    timer.start_record("face_detection")
    output = face_detection_engine(full_image).detach().cpu().numpy()
    # print("output_shape:", output.shape)
    timer.end_record("face_detection")
    timer.start_record("face_detection_postprocess")
    predictions = demo_postprocess(output, input_shape)[0]
    # print("predictions shape:", predictions.shape)
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

def detect_landmark_trt(img, face, detect_landmark_engine, timer) -> np.ndarray:
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
        
    timer.start_record("landmark_detection")
    input = torch.Tensor(input[None, :, :, :]).cuda()
    _, landmark = detect_landmark_engine(input)
    landmark = landmark.detach().cpu().numpy()
    timer.end_record("landmark_detection")

    pre_landmark = landmark.reshape(-1, 2) * [size, size]
    landmark_on_cropped = pre_landmark.copy()
    pre_landmark -= [edx1, edy1]
    pre_landmark[:, 0] += x1
    pre_landmark[:, 1] += y1

    return pre_landmark, landmark_on_cropped, cropped

def estimate_gaze_trt(img, landmark, gaze_estimation_engine, timer) -> np.ndarray:
    timer.start_record("gaze_estimation_preprocess")
    rvec, tvec = estimateHeadPose(landmark)
    data = normalizeDataForInference(img, rvec, tvec)
    timer.end_record("gaze_estimation_preprocess")
    leye_image, reye_image, face_image, R = data

    leye_image = np.ascontiguousarray(leye_image, dtype=np.float32) / 255.0
    leye_image = np.transpose(np.expand_dims(leye_image, 0), (0,3,1,2))
    leye_image = torch.Tensor(leye_image).cuda()

    reye_image = np.ascontiguousarray(reye_image, dtype=np.float32) / 255.0
    reye_image = np.transpose(np.expand_dims(reye_image, 0), (0,3,1,2))
    reye_image = torch.Tensor(reye_image).cuda()
    
    face_image = np.ascontiguousarray(face_image, dtype=np.float32) / 255.0
    face_image = np.transpose(np.expand_dims(face_image, 0), (0,3,1,2))
    face_image = torch.Tensor(face_image).cuda()

    print("leye, reye, face shape:", leye_image.shape, reye_image.shape, face_image.shape)

    timer.start_record("gaze_estimation")
    pred_pitchyaw_aligned = gaze_estimation_engine(leye_image, reye_image, face_image).detach().cpu().numpy()
    pred_pitchyaw_aligned = pred_pitchyaw_aligned[0]
    timer.end_record("gaze_estimation")
    print("output shape:", pred_pitchyaw_aligned.shape)

    pred_pitchyaw_aligned = np.deg2rad(pred_pitchyaw_aligned).tolist()
    pred_vec_aligned = euler_to_vec(*pred_pitchyaw_aligned)
    pred_vec_cam = np.dot(np.linalg.inv(R), pred_vec_aligned)
    pred_vec_cam /= np.linalg.norm(pred_vec_cam)
    pred_pitchyaw_cam = np.array(vec_to_euler(*pred_vec_cam))
    return pred_pitchyaw_cam, rvec, tvec


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
    h, w = img.shape[:2]
    x1, y1 = start_point
    x2, y2 = find_edge_intersection(w, h, start_point, end_point)

    print("H:", h, "W:", w)

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
    while 0<=x<w and 0<=y<h:
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
    
    gaze_points = np.array([(x,y) for x,y in zip(x_vals, y_vals)])

    # for x, y in zip(x_vals, y_vals):
    #     cv2.circle(img, (x,y), 1, (255, 0, 0), thickness=1)

    mask = np.zeros((h, w))
    mask[y_vals, x_vals] = 1

    return gaze_points, mask.astype(bool)