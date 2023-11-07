ONNXPATH="../../engines/vit/onnx/decoder_64_80_dympoints.onnx"
ENGINEPATH="../../engines/vit/decoder_64_80_dympoints.engine"
echo "export efficientvit sam decoder >>>"

python -m make_decoder \
    --checkpoint vit/l1.pt \
    --output $ONNXPATH \
    --model-type l1 \
    --opset 11
echo "--- base onnx file creation complete ---"

polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
echo "--- constant folding complete ---"

echo "--- creating trt engine ---"
/home/nicole/TensorRT-8.4.3.1/bin/trtexec \
    --onnx=$ONNXPATH \
    --saveEngine=$ENGINEPATH \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:64x1x2,point_labels:64x1 \
    --maxShapes=point_coords:80x10x2,point_labels:80x10 \
    --verbose