FACE_ONNX_PATH="face_detection.onnx"
FACE_ENGINE_PATH="../../engines/gaze/face_detection_fp16.engine"

LANDMARK_ONNX_PATH="landmark_detection.onnx"
LANDMARK_ENGINE_PATH="../../engines/gaze/landmark_detection_fp16.engine"

GAZE_ONNX_PATH="gaze_estimation.onnx"
GAZE_ENGINE_PATH="../../engines/gaze/gaze_estimation_fp16.engine"

polygraphy surgeon sanitize --fold-constants $FACE_ONNX_PATH -o $FACE_ONNX_PATH
echo "--- constant folding (face) complete ---"

polygraphy surgeon sanitize --fold-constants $LANDMARK_ONNX_PATH -o $LANDMARK_ONNX_PATH
echo "--- constant folding (landmark) complete ---"

polygraphy surgeon sanitize --fold-constants $GAZE_ONNX_PATH -o $GAZE_ONNX_PATH
echo "--- constant folding (gaze) complete ---"

echo "--- creating face trt engine ---"
trtexec \
    --onnx=$FACE_ONNX_PATH \
    --saveEngine=$FACE_ENGINE_PATH \
    --fp16

echo "--- creating landmark trt engine ---"
trtexec \
    --onnx=$LANDMARK_ONNX_PATH \
    --saveEngine=$LANDMARK_ENGINE_PATH \
    --fp16

echo "--- creating gaze trt engine ---"
trtexec \
    --onnx=$GAZE_ONNX_PATH \
    --saveEngine=$GAZE_ENGINE_PATH \
    --fp16
