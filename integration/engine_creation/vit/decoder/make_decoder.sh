ONNXPATH="../../../engines/vit/onnx/decoder_k9_fp32_trt8.6_unstacked_l0_opset11.onnx"
# ENGINEPATH="../../../engines/vit/decoder_k9_fp32_trt8.6_unstacked_l0.engine"
# echo "export efficientvit sam decoder >>>"

python -m make_decoder \
    --checkpoint l0.pt \
    --output $ONNXPATH \
    --model-type l0 \
    --opset 11
echo "--- base onnx file creation complete ---"

# polygraphy surgeon sanitize --fold-constants $ONNXPATH -o $ONNXPATH
# echo "--- constant folding complete ---"

# echo "--- creating trt engine ---"
# trtexec \
#     --onnx=$ONNXPATH \
#     --saveEngine=$ENGINEPATH