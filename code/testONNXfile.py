import cv2
import numpy as np
import onnxruntime as ort

# Function to load the ONNX model
def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# Function to preprocess the input frame (resize, normalize, etc.)
def preprocess_frame(frame):
    # Resize the frame to the expected input size for the model (assuming 640x640 for this example)
    resized = cv2.resize(frame, (640, 640))
    
    # Normalize the image (assuming mean and std values of [0.5] for simplicity, adjust based on your model)
    normalized = resized / 255.0
    normalized = np.transpose(normalized, (2, 0, 1))  # Change to CxHxW
    normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
    normalized = normalized.astype(np.float32)  # Ensure the correct type
    return resized, normalized  # Return resized image and the normalized data

# Function to postprocess the model output (extract boxes, classes, and confidence)
def postprocess_output(output, img_width, img_height):
    # Extract raw bounding box data from the output
    boxes = output[0]  # Assuming the output is a list or array
    bboxes = []
    
    # Loop through the boxes and extract the bounding boxes
    for i in range(boxes.shape[2]):  # Iterate over detections (8400 boxes)
        # Extract the 6 values for each box (x1, y1, x2, y2, confidence, class_id)
        box = boxes[0, :, i]  # Get each box (6 values per box)
        
        # Extract coordinates and confidence from the box
        x1, y1, x2, y2, confidence, class_id = box
        
        # Filter out low-confidence predictions (thresholding)
        if confidence > 0.5:
            # Rescale the bounding box coordinates to the original image size
            scale_x = img_width / 640  # Resize width scale factor
            scale_y = img_height / 640  # Resize height scale factor
            
            # Rescale the bounding box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Clip bounding box coordinates to be within the image dimensions
            x1, y1, x2, y2 = max(0, min(x1, img_width)), max(0, min(y1, img_height)), \
                               max(0, min(x2, img_width)), max(0, min(y2, img_height))
            
            bboxes.append((x1, y1, x2, y2, confidence, class_id))
    
    return bboxes

# Function to run inference on a video and display results
def run_inference_on_video(model_path, video_path):
    # Load ONNX model
    session = load_onnx_model(model_path)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the frame's width and height
        img_height, img_width = frame.shape[:2]
        
        # Preprocess the frame
        resized, input_data = preprocess_frame(frame)
        
        # Run the inference on the frame
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
        
        # Post-process the outputs to extract bounding boxes
        boxes = postprocess_output(outputs, img_width, img_height)
        
        # Draw the bounding boxes on the frame
        for (x1, y1, x2, y2, confidence, class_id) in boxes:
            # Draw rectangle (bounding box)
            cv2.rectangle(frame, (int(x1 + (x2 / 2)), int(y1 + (y2 / 2))), (int(x1 - (x2 / 2)), int(y1 - (y2 / 2))), (0, 255, 0), 2)
            # Display the confidence
            cv2.putText(frame, f'Conf: {confidence:.2f}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show the frame with bounding boxes
        cv2.imshow("Inference Output", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Run inference on a given video and ONNX model
if __name__ == "__main__":
    model_path = "model.onnx"  # Replace with the path to your ONNX model
    video_path = "Test1Cropped.mp4"  # Replace with the path to your video file
    run_inference_on_video(model_path, video_path)