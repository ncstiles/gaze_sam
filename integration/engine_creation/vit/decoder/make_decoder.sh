ONNXPATH="../../../engines/vit/onnx/box_decoder.onnx"
ENGINEPATH="../../../engines/vit/box_decoder.engine"
echo "export efficientvit sam decoder >>>"
 
python -m make_decoder \
    --checkpoint l0.pt \
    --output $ONNXPATH \
    --model-type l0 \
    --opset 12
echo "--- base onnx file creation complete ---"

polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
echo "--- constant folding complete ---"

echo "--- creating trt engine ---"
trtexec \
    --onnx=$ONNXPATH \
    --saveEngine=$ENGINEPATH \
    --minShapes=boxes:1x1x4 \
    --optShapes=boxes:3x1x4 \
    --maxShapes=boxes:10x1x4
    # --minShapes=point_coords:1x1x2,point_labels:1x1 \
    # --optShapes=point_coords:1x1x4,point_labels:1x1 \
    # --maxShapes=point_coords:32x1x4,point_labels:32x1 \
