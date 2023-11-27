timestamp=$(date +"%Y%m%d_%H%M%S")

python3 build_int8_decoder.py \
    --onnx engines/vit/onnx/decoder_k9_fp32_l0_fixed_size_stacked_output_iou_last_tensorrt_8.6_opset17.onnx \
    --engine engines/vit/decoder_k9_int8_latency_test_stacked_iou_last_trt8.6.engine \
    --precision int8 \
    --calib_input ../../fiftyone/coco-2017/train/data \
    --calib_cache decoder_5k_images_bs_1_stacked_iou_last_trt8.6.cache \
    --calib_batch_size 1 \
    --calib_num_images 5000 \
    --encoder engines/vit/encoder_k9_fp32_trt8.6.engine
    # --calib_cache decoder_5k_images_bs_1_20231118_142823.cache \
    # --onnx engines/vit/onnx/decoder_k9_fp32_fixed_size_for_latency_test.onnx \
