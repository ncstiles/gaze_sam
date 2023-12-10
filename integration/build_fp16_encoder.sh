python build_fp16_encoder.py \
    --onnx engines/vit/onnx/encoder_k9_fp32.onnx \
    --engine engines/vit/encoder_fp16_trt8.6.engine \
    # --verbose