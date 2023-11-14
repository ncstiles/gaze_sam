python3 build_int8_decoder.py \
    --onnx engines/vit/onnx/k9/encoder_k9_fp32.onnx \
    --engine engines/vit/encoder_k9_int8_128gum.engine \
    --precision int8 \
    --calib_input ../base_imgs/all_gum \
    --calib_cache encoder_person_dataset_128_gum.cache \
    --calib_batch_size 128 \
    --calib_num_images 128
