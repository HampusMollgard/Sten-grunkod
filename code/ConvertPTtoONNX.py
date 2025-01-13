import torch

# Step 1: Load the checkpoint
checkpoint = torch.load("best.pt", map_location="cpu")  # Load the checkpoint file

# Step 2: Extract the model
model = checkpoint["model"]  # Use the 'model' key
# Convert the model to float32
model.float()
model.eval()  # Set the model to evaluation mode
print("Model successfully loaded!")

# Step 3: Define a dummy input that matches your model's expected input size
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust size to match your input (e.g., YOLO models use 640x640)

# Step 4: Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,  # Store the trained parameter weights in the model file
    opset_version=11,  # Adjust the ONNX opset version if needed
    do_constant_folding=True,  # Optimize the model
    input_names=["images"],  # Name of the input
    output_names=["output"],  # Name of the output
)

print("ONNX model exported successfully!")