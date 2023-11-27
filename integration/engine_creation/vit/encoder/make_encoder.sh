ONNXPATH="../../../engines/vit/onnx/encoder_k9_fp32.onnx"
ENGINEPATH="../../../engines/vit/encoder_g0_fp32_trt8.6.engine"

echo "export efficientvit sam encoder (no built-in preprocessing)>>>"

python -m make_encoder \
    --checkpoint l1.pt \
    --output $ONNXPATH \
    --model-type l1 \
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

