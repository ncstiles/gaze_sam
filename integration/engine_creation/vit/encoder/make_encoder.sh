ONNXPATH="../../../engines/vit/onnx/encoder_fp32.onnx"
ENGINEPATH="../../../engines/vit/encoder_fp32.engine"

echo "export efficientvit sam encoder (no built-in preprocessing)>>>"

python -m make_encoder \
    --checkpoint l0.pt \
    --output $ONNXPATH \
    --model-type l0 \
    --opset 12 \
    
echo "--- base onnx file creation complete ---"

polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
echo "--- constant folding complete ---"

python3 remove_resize_cubic.py --onnx_path $ONNXPATH
echo "--- resize cubic interpolations casted to linear complete ---"

echo "--- creating trt engine ---"
trtexec \
    --onnx=$ONNXPATH \
    --saveEngine=$ENGINEPATH \
    # --fp16

