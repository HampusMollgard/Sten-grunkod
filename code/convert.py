import onnx

# Load the ONNX model
onnx_model = onnx.load("best.onnx")

# Print the model's input shapes
for input in onnx_model.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])