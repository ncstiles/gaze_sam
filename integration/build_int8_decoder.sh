timestamp=$(date +"%Y%m%d_%H%M%S")

python3 build_int8_decoder.py \
    --onnx engines/vit/onnx/decoder_k5_fp32_fixed_size.onnx \
    --engine engines/vit/decoder_k9_int8_1gum.engine \
    --precision int8 \
    --calib_input ../base_imgs/all_gum \
    --calib_cache del_$timestamp.cache \
    --calib_batch_size 1 \
    --calib_num_images 1

# python3 build_int8_decoder.py \
#     --onnx engines/vit/onnx/decoder_k5_fp32_fixed_size.onnx \
#     --engine engines/vit/decoder_k9_fp32_using_int8_script.engine \
#     --precision fp32 \
