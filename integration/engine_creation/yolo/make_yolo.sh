ONNXPATH="yolo_fp32.onnx"
ENGINEPATH="../../engines/yolo/yolo_fp32.engine"
echo "export yolo nas engine >>>"

python -m make_yolo \
    --model-name $ONNXPATH
echo "--- base onnx file creation + input type conversion complete ---"

polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
echo "--- constant folding complete ---"

echo "--- creating trt engine ---"
trtexec \
    --onnx=$ONNXPATH \
    --saveEngine=$ENGINEPATH \
    # --fp16
