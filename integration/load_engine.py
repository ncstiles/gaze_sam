from torch2trt import TRTModule
import tensorrt as trt


# --- vit engines ---

def load_image_encoder_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["input_image"],
        output_names=["image_embeddings"]
    )

    return image_encoder_trt

def load_mask_decoder_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    mask_decoder_trt = TRTModule(
        engine=engine,
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input"
        ],
        # output_names=[
        #     "stacked_output"
        # ]
        output_names=[
            "iou_predictions",
            "low_res_masks"
        ]
    )

    return mask_decoder_trt

# --- gaze engines ---

def load_face_detection_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["full_image"],
        output_names=["bbox_det"]
    )

    return image_encoder_trt

def load_landmark_detection_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["face_image"],
        output_names=["headpose_feature", "facial_landmark"]
    )

    return image_encoder_trt


def load_gaze_estimation_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["left_eye", "right_eye", "face"],
        output_names=["gaze_pitchyaw"]
    )

    return image_encoder_trt

# --- yolo engines ---

def load_yolo_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    trtengine = TRTModule(
        engine=engine,
        input_names=["onnx::Cast_0"],
        output_names=["num_dets", "det_boxes", "det_scores", "det_classes"]
    )

    return trtengine