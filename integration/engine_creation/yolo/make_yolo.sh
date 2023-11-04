ONNXPATH="yolo.onnx"
ENGINEPATH="yolo.engine"
echo "export yolo nas engine >>>"

python -m make_yolo \
    --model-name $ONNXPATH
echo "--- base onnx file creation + input type conversion complete ---"

polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
echo "--- constant folding complete ---"

echo "--- creating trt engine ---"
/home/nicole/TensorRT-8.4.3.1/bin/trtexec \
    --onnx=$ONNXPATH \
    --saveEngine=$ENGINEPATH \
