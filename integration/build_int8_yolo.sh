timestamp=$(date +"%Y%m%d_%H%M%S")

python3 build_int8_yolo.py \
    --onnx engines/yolo/onnx/yolo_k9.onnx \
    --engine engines/yolo/yolo_k9_int8_trt8.6.engine \
    --precision int8 \
    --calib_input ../../fiftyone/coco-2017/train/data \
    --calib_cache yolo_5k_people_500bs_trt8.6_cache \
    --calib_batch_size 500 \
    --calib_num_images 5000

