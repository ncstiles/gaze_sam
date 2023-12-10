python build_fp16_decoder.py \
    --onnx engines/vit/onnx/decoder_k9_fp32_trt8.6_unstacked_l0_opset11.onnx \
    --engine engines/vit/decoder_fp16_k9_unstacked_l0_opset11.engine \
    --encoder engines/vit/encoder_k9_fp32_trt8.6.engine \
    # --onnx engines/vit/onnx/decoder_k9_fp32_trt8.6_unstacked_l1.onnx \
    # --verbose
