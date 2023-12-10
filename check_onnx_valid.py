import onnx


def is_onnx_model_valid(path):
    model = onnx.load(path)
    try:
        res = onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("e:", e)
        return False
    return True


print(is_onnx_model_valid('saved_models/updated_yolo.onnx'))