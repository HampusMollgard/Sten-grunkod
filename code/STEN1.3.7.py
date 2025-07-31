import cv2
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import numpy as np
import requests
import supervision as sv
import math
import time

devMode = False

# Create OpenCV window for displaying the results
cv2.namedWindow("Processed camera", cv2.WINDOW_NORMAL)

prevEntryPoint = None

min_distance = 100

class_names = ["90 degree", "Entry_point", "Green", "T-intersection", "X-intersection"]
class_names_evaczone = ["Dead victim", "Live victim"]

if not devMode:
    from picamera2 import Picamera2
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams,
                            FormatType)
    import serial

    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

    picam2 = Picamera2(1)
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 640)}))
    #picam2.set_controls({'ExposureTime': 10000})
    picam2.start()

    picam2down = Picamera2(0)
    picam2down.configure(picam2.create_preview_configuration(main={"size": (640, 640)}))
    #picam2.set_controls({'ExposureTime': 10000})
    picam2down.start()

    output_filename = '/home/malte/film1.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recordingFPS = 40
    recording_size = (640, 640)
    out = cv2.VideoWriter(output_filename, fourcc, recordingFPS, recording_size)

    target = VDevice()

    hef_path = '/home/malte/Downloads/best10.hef'
    hef = HEF(hef_path)

    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info.shape


    hef_path2 = '/home/malte/Downloads/balls1.hef'
    hef2 = HEF(hef_path2)

    configure_params2 = ConfigureParams.create_from_hef(hef=hef2, interface=HailoStreamInterface.PCIe)
    network_groups2 = target.configure(hef2, configure_params2)
    network_group2 = network_groups2[0]
    network_group_params2 = network_group2.create_params()

    input_vstreams_params2 = InputVStreamParams.make(network_group2, format_type=FormatType.FLOAT32)
    output_vstreams_params2 = OutputVStreamParams.make(network_group2, format_type=FormatType.FLOAT32)
    
    input_vstream_info2 = hef2.get_input_vstream_infos()[0]
    output_vstream_info2 = hef2.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info2.shape

if devMode:
    model = YOLO('best-4.torchscript')
    video_path = "TestRun3.mp4"  # Update with your video path
    cap = cv2.VideoCapture(video_path)  

# Initialize a variable to store the time of the last call
last_time = None

def displayGUI(img1_=None, img2_=None, img3_=None):
    global last_time
    # Use static variables to store the previous images
    if not hasattr(displayGUI, "prev_img1"):
        displayGUI.prev_img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    if not hasattr(displayGUI, "prev_img2"):
        displayGUI.prev_img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    if not hasattr(displayGUI, "prev_img3"):
        displayGUI.prev_img3 = np.zeros((640, 640, 3), dtype=np.uint8)

    # Use the previous image if no image is provided
    img1_ = img1_ if img1_ is not None else displayGUI.prev_img1
    img2_ = img2_ if img2_ is not None else displayGUI.prev_img2
    img3_ = img3_ if img3_ is not None else displayGUI.prev_img3

    # Update the previous images
    displayGUI.prev_img1 = img1_
    displayGUI.prev_img2 = img2_
    displayGUI.prev_img3 = img3_

    height, width, _ = img1_.shape

    # Resize images if necessary
    img2_ = cv2.resize(img2_, (640, 640))

    # Calculate FPS
    current_time = time.time()
    if last_time is not None:
        fps = 1 / (current_time - last_time)
    else:
        fps = 0  # First frame, FPS cannot be calculated
    last_time = current_time    

    # Create a blank image with 2x2 grid size
    combined_img = np.zeros((height*2, width*2, 3), dtype=np.uint8)

    # Place the images in the corners
    combined_img[0:height, 0:width] = img1_
    if len(img2_.shape) == 2:  # Check if img2_ is a mask (grayscale image)
        combined_img[0:height, width:width*2] = cv2.cvtColor(img2_, cv2.COLOR_GRAY2BGR)
    else:  # Otherwise, assume it's already a BGR image
        combined_img[0:height, width:width*2] = img2_
    combined_img[height:height*2, 0:width] = img3_

    # Use a static variable to store FPS values
    if not hasattr(displayGUI, "fps_history"):
        displayGUI.fps_history = []

    # Append the current FPS to the history
    displayGUI.fps_history.append(fps)

    # Keep only the last 100 FPS values
    if len(displayGUI.fps_history) > 100:
        displayGUI.fps_history.pop(0)

    # Create a graph area that takes up the entire bottom-right corner
    graph_height = height
    graph_width = width
    graph_img = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

    # Normalize FPS values for graph scaling
    max_fps = max(displayGUI.fps_history) if displayGUI.fps_history else 1
    if max_fps == 0:
        max_fps = 1
    normalized_fps = [int((value / max_fps) * (graph_height - 10)) for value in displayGUI.fps_history]

    # Draw the FPS graph
    for i in range(1, len(normalized_fps)):
        cv2.line(graph_img, 
                 (int((i - 1) * (graph_width / 100)), graph_height - normalized_fps[i - 1]), 
                 (int(i * (graph_width / 100)), graph_height - normalized_fps[i]), 
                 (0, 255, 0), 2)

    # Add FPS text to the graph
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(graph_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Place the graph in the bottom-right corner
    combined_img[height:height*2, width:width*2] = graph_img

    # Display the combined image in a named window
    cv2.imshow('Combined Image', combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

def Inference(frame_, model_index):
    if model_index == 0:
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                if frame_ is None:
                    print("Failed to capture frame")
                    return None

                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                image_input = frame_rgb.astype(np.float32)
                image_input = np.expand_dims(image_input, axis=0)
                input_data = {input_vstream_info.name: image_input}
                infer_results = infer_pipeline.infer(input_data)
                detections = infer_results['yolov8n/yolov8_nms_postprocess'][0] 
                return detections
    else:
        with InferVStreams(network_group2, input_vstreams_params2, output_vstreams_params2) as infer_pipeline2:
            with network_group2.activate(network_group_params2):
                if frame_ is None:
                    print("Failed to capture frame")
                    return None

                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                image_input = frame_rgb.astype(np.float32)
                image_input = np.expand_dims(image_input, axis=0)
                input_data = {input_vstream_info2.name: image_input}
                infer_results = infer_pipeline2.infer(input_data)
                detections = infer_results['yolov8n/yolov8_nms_postprocess'][0] 
                return detections
    
def get_picture(dev_, camera_index):
    ret_ = False 
    if dev_:
        ret_, frame_ = cap.read()
    else:
        if camera_index == 0:
            frame_ = picam2.capture_array()
        else:
            frame_ = picam2down.capture_array()
        ret_ = True
    if frame_.shape[-1] == 4:
        frame_ = frame_[:, :, :3]
    return ret_, frame_

def math_sign(value):
    return (value > 0) - (value < 0)

def send_data(val1, val2, val3):
    message = f"{val1},{val2},{val3}\n"  # Send as CSV with newline
    ser.write(message.encode())  

def send_speciall_data(val):
    message = f"{val}\n"  # Send as CSV with newline
    ser.write(message.encode())  

def runMotors(radius, wanted_speed_outside):
    reversed = 1
    if radius > 0:
        reversed = 0
    if not devMode:
        send_data(round(abs(radius)), wanted_speed_outside, reversed)

def driveTowardsPoint(cX, cY, frame_height, frame_width):
    x_point, y_point = cX, cY
    lower_middle_x = frame_width / 2

    # Calculate the differences
    dx = x_point - lower_middle_x
    dy = frame_height - y_point

    # Calculate the angle in degrees
    DrivingAngle = math.degrees(math.atan2(dy, dx))
    if DrivingAngle > 90:
        DrivingAngle = -180 + DrivingAngle
            
    if DrivingAngle > 0:
        DrivingAngle = abs(0.01*abs(DrivingAngle)+pow(1.1, abs(DrivingAngle) - 45))
    else:
        DrivingAngle = -abs(0.01*abs(DrivingAngle)+pow(1.1, abs(DrivingAngle) - 45))

    runMotors(DrivingAngle, 10)

def normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width):
    #forward = get_distance_forward(ser, send_speciall_data)
    
    #print(f"Forward distance: {forward}")
    #if forward and forward > 0 and forward < 50:
        #drive_round_object()
        #return


    vertical_distance = abs(300 - bottom_most_point[1])

    # Calculate the width to maintain the aspect ratio
    aspect_ratio = frame.shape[1] / frame.shape[0]
    width = int(vertical_distance * aspect_ratio)

        # Calculate the cropping coordinates
    x_start = int(max(0, (bottom_most_point[0] + 70) - width // 2))
    x_end = int(min(frame.shape[1], (bottom_most_point[0] + 70) + width // 2))
    y_start = int(max(0, bottom_most_point[1] - vertical_distance))
    y_end = int(bottom_most_point[1])

    cropped_mask = mask[y_start:y_end, x_start:x_end]
    mask_middle = np.mean(np.nonzero(cropped_mask), axis=1).astype(int)

    cv2.circle(annotated_image, (x_start + mask_middle[1], y_start + mask_middle[0]), 10, (255, 255, 255), -1)

    top_most_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    cv2.circle(annotated_image, top_most_point, 15, (255, 0, 0), -1)

    if top_most_point[1] < frame_height / 2:
        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width)
    else:
        if top_most_point[1] < frame_height - 170:
            print("Gap detected, driving towards end")
            driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width)
        else:
            print("Gap")
            runMotors(200, 10)

            while top_most_point[1] > frame_height - 170:
                ret, frame = get_picture(devMode, 0)
                if not ret:
                    print("End of video or cannot read frame.")
                    break  # Exit loop when video ends
                annotated_image = frame.copy()  
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 60])
                mask = cv2.inRange(hsv_frame, lower_black, upper_black)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bounding_box_area = w * h
                    image_area = frame.shape[0] * frame.shape[1]

                    if bounding_box_area >= 0.05 * image_area:
                        top_most_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                        cv2.circle(annotated_image, top_most_point, 15, (255, 0, 0), -1)
                        print(f"Topmost point: {top_most_point}")

                displayGUI(annotated_image)
runMotors(0, 0)
time.sleep(1.5)

def drive_round_object():
    runMotors(0, 0)
    time.sleep(1)
    largest_bounding_box = 5000  # Maximum bounding box area
    turningRight = False
    forwardRight, forward, forwardLeft, rearRight = get_distances_full(ser, send_speciall_data)
    print(rearRight)
    if rearRight < 300:
        print("turning left")
        runMotors(-0.1, 10)
        pointTurn(devMode, get_picture, 90)
        runMotors(30, 10)
    else:
        print("turning right")
        turningRight = True
        runMotors(0.1, 10)
        pointTurn(devMode, get_picture, 90)
        runMotors(-30, 10)
    
    while True:
        ret, frame = get_picture(devMode, 0)
        if not ret:
            print("End of video or cannot read frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Black mask
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(annotated_image, [largest_contour], -1, (0, 255, 0), 2)

            # Calculate the bounding box area of the largest contour
            _, _, w, h = cv2.boundingRect(largest_contour)
            bounding_box_area = w * h

            # Break the loop if the bounding box area exceeds the threshold
            if bounding_box_area > largest_bounding_box:
                print("Largest black contour exceeds the bounding box threshold. Exiting loop.")
                break
    if turningRight:
        runMotors(-0.1, 10)
        pointTurn(devMode, get_picture, 90)
    else:
        runMotors(0.1, 10)
        pointTurn(devMode, get_picture, 90)

def pointTurn(devMode, get_picture, angle):
    start_time = time.time()
    while time.time() - start_time < 0.9 * (angle / 90):
        ret, tempframe = get_picture(devMode, 0)
        if not ret:
            print("End of video or cannot read frame.")
            break
        displayGUI(annotated_image)
    return

def forwardDrive(devMode, get_picture, drivingTime):
    start_time = time.time()
    while time.time() - start_time < drivingTime:
        ret, tempframe = get_picture(devMode, 0)
        if not ret:
            print("End of video or cannot read frame.")
            break
        displayGUI(annotated_image)
    return

while True:
    start = time.time()
    ret, frame = get_picture(devMode, 0)
    if not ret:
        print("End of video or cannot read frame.")
        break  # Exit loop when video ends
    
    annotated_image = frame.copy()
    inference_image = frame.copy()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Black mask
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    mask = cv2.inRange(hsv_frame, lower_black, upper_black)

    lower_red2 = np.array([100, 150, 100])
    upper_red2 = np.array([130, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Find contours in the red mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if red_contours:
        largest_red_contour = max(red_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_red_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if w >= 0.9 * frame.shape[1]:
            runMotors(0, 0)
            time.sleep(6)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(annotated_image, [largest_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(annotated_image, (cX, cY), 15, (255, 255, 255), -1)

        # Find the bottom-most point
        bottom_most_point = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        cv2.circle(annotated_image, bottom_most_point, 15, (0, 0, 255), -1)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    inference_start = time.time()
    detections = Inference(inference_image, 0)
    inference_time = time.time() - inference_start

    objectDetected = False

    for idx, det in enumerate(detections):
        if len(det) == 0:
            continue
        for thing in det:
            
            x_min, y_min, x_max, y_max, confidence = thing
        
            if confidence > 0.2:  # Filter low-confidence detections
                if class_names[idx] != "Green":  # Ignore class with index 1
                    objectDetected = True
                image_width, image_height = frame_height, frame_width
                new_width, new_height = image_height, image_width
            
                x_new_min = y_min
                x_new_max = y_max
                y_new_min = new_width - x_max
                y_new_max = new_width - x_min
            
                y_new_min = new_height - y_new_min
                y_new_max = new_height - y_new_max
            
                x_min = int(x_new_min * image_width)
                y_min = int(y_new_min * image_height)
                x_max = int(x_new_max * image_width)
                y_max = int(y_new_max * image_height)
                
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
                label = f"{class_names[idx]} {(confidence * 100):.0f}%"
                cv2.putText(annotated_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_image, str(time.time() - start), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    if objectDetected:

        closest_corner = None
        min_distance = float('inf')
        
        closest_index = -1  # Initialize closest index
        closest_corners = None

        for idx, det in enumerate(detections):
            if len(det) == 0:
                continue
            for idx2, thing in enumerate(det):
                if class_names[idx] == "Green":  # Ignore class with index 1
                    continue

                x_min, y_min, x_max, y_max, confidence = thing
        
                if confidence > 0.2:  # Filter low-confidence detections
                    new_width, new_height = image_height, image_width
                
                    x_new_min = y_min
                    x_new_max = y_max
                    y_new_min = new_width - x_max
                    y_new_max = new_width - x_min
                
                    y_new_min = new_height - y_new_min
                    y_new_max = new_height - y_new_max
                
                    x_min = int(x_new_min * image_width)
                    y_min = int(y_new_min * image_height)
                    x_max = int(x_new_max * image_width)
                    y_max = int(y_new_max * image_height)
                
                    corners = [
                        (x_min, y_min),  # Top-left
                        (x_max, y_min),  # Top-right
                        (x_min, y_max),  # Bottom-left
                        (x_max, y_max)   # Bottom-right
                    ]
                    
                    for corner in corners:
                        distance = math.sqrt((corner[0] - bottom_most_point[0]) ** 2 + (corner[1] - bottom_most_point[1]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_corner = corner
                            closest_corners = corners
                            closest_index = (idx, idx2)
        
        if closest_corner:
            cv2.circle(annotated_image, (int(closest_corner[0]), int(closest_corner[1])), 10, (255, 255, 0), -1)

            # Calculate vertical distance
            vertical_distance = abs(closest_corner[1] - bottom_most_point[1])

            # Calculate the width to maintain the aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            width = int(vertical_distance * aspect_ratio)

            # Calculate the cropping coordinates
            x_start = int(max(0, (bottom_most_point[0] + 70) - width // 2))
            x_end = int(min(frame.shape[1], (bottom_most_point[0] + 70) + width // 2))
            y_start = int(max(0, bottom_most_point[1] - vertical_distance))
            y_end = int(bottom_most_point[1])

            # Only crop the mask if the height is greater than 80
            if (y_end - y_start) > 100:
                # Crop the mask
                cropped_mask = mask[y_start:y_end, x_start:x_end]
                if cropped_mask is not None:
                    mask_middle = np.mean(np.nonzero(cropped_mask), axis=1).astype(int)
                    cv2.circle(annotated_image, (x_start + mask_middle[1], y_start + mask_middle[0]), 10, (0, 255, 255), -1)
                    driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width)
            else:
                closest_class = closest_index[0]
                if class_names[closest_class] == "90 degree":  # 90 Degree
                    x_start = closest_corners[0][0]
                    x_end = closest_corners[1][0]
                    y_start = min(closest_corners[0][1], closest_corners[2][1])
                    y_end = max(closest_corners[0][1], closest_corners[2][1])

                    # Calculate the center point
                    center_x = (x_start + x_end) // 2
                    center_y = (y_start + y_end) // 2
                    cv2.circle(annotated_image, (center_x, center_y), 10, (0, 0, 255), -1)
                    cropped_mask = mask[y_start:y_end, x_start:x_end]

                    # Define the rectangle inside the cropped_mask
                    inner_x_start = max(0, 10)
                    inner_x_end = min(cropped_mask.shape[1], cropped_mask.shape[1] - 10)
                    inner_y_start = max(0, 10)
                    inner_y_end = min(cropped_mask.shape[0], cropped_mask.shape[0] - 10)

                    # Remove everything inside the rectangle
                    cropped_mask[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = 0

                    # Find contours in the modified cropped mask
                    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Sort contours by area in descending order
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    # Get the coordinates of the two largest areas
                    if len(contours) >= 2:
                        area1 = cv2.boundingRect(contours[0])
                        area2 = cv2.boundingRect(contours[1])

                        bottom_middle = (frame_width // 2, frame_height - 1)

                        # Draw circles at the areas on annotated_image with offset
                        point1 = (x_start + (area1[0] + area1[2] // 2), y_start + (area1[1] + area1[3] // 2))
                        point2 = (x_start + (area2[0] + area2[2] // 2), y_start + (area2[1] + area2[3] // 2))
                        cv2.circle(annotated_image, point1, 10, (0, 255, 255), -1)
                        cv2.circle(annotated_image, point2, 10, (0, 255, 255), -1)

                        # Find which of the points is closest to the bottom of the frame (y = frame_height - 1)
                        distance1 = math.sqrt((point1[0] - bottom_most_point[0]) ** 2 + (point1[1] - (frame_height - 1)) ** 2)
                        distance2 = math.sqrt((point2[0] - bottom_most_point[0]) ** 2 + (point2[1] - (frame_height - 1)) ** 2)
                        
                        if distance1 < distance2:
                            closest_point = point1
                            furthest_point = point2
                        else:
                            closest_point = point2
                            furthest_point = point1
                            
                        cv2.circle(annotated_image, closest_point, 15, (255, 0, 0), -1)

                        closest_point = bottom_middle

                        # Calculate the angle between the two points
                        vector1 = (closest_point[0] - center_x, closest_point[1] - center_y)
                        vector2 = (furthest_point[0] - center_x, furthest_point[1] - center_y)
                        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
                        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
                        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
                        angle = math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

                        # Determine direction (right or left)
                        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
                        direction = "right" if cross_product < 0 else "left"

                        angle = 180 - angle 

                        cv2.circle(annotated_image, closest_point, 15, (255, 0, 0), -1)

                        # Calculate the angle between a vertical line and the line from the bottom middle to center_x, center_y
                        vertical_vector = (0, -1)  # Vertical line pointing upwards
                        line_vector = (center_x - bottom_middle[0], center_y - bottom_middle[1])
                        dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                        magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                        magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                        vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                        if center_x < frame_width // 2:
                            vertical_angle = -vertical_angle

                        # Calculate the length between bottom_middle and center_x / center_y
                        distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - bottom_middle[1]) ** 2)

                        displayGUI(annotated_image)

                        runMotors(math_sign(vertical_angle) * 0.1, 10)
                        pointTurn(devMode, get_picture, vertical_angle)
                        runMotors(100, 10)
                        forwardDrive(devMode, get_picture, (0.6 * (distance_to_center / 200)))
                        if cross_product < 0:
                            runMotors(0.1, 10)
                            pointTurn(devMode, get_picture, angle)
                        else:
                            runMotors(-0.1, 10)
                            pointTurn(devMode, get_picture, angle)
                        runMotors(100, -10)
                        forwardDrive(devMode, get_picture, 0.2)
                        runMotors(0, 0)

                        annotated_image = None
                    else:
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
                elif class_names[closest_class] == "Entry_point":  # Green
                    x_start = closest_corners[0][0]
                    x_end = closest_corners[1][0]
                    object_width = abs(x_end - x_start)

                    y_start = min(closest_corners[0][1], closest_corners[2][1])
                    y_end = max(closest_corners[0][1], closest_corners[2][1])
                    
                    object_middle_y = (y_start + y_end) // 2  # Calculate the middle Y-coordinate of the object
                    
                    if object_width >= 0.9 * frame_width:  # Check width and Y-coordinate
                        runMotors(-100, 10)
                        time.sleep(3)
                        runMotors(0, 0)
                        balls = ["Live victim", "Live victim", "Dead victim"]
                        #balls = ["Live victim"]
                        ballIndex = 0
                        
                        while ballIndex < len(balls): 
                            ret, frame = get_picture(devMode, 1)
                            if not ret:
                                print("End of video or cannot read frame.")
                                break  # Exit loop when video ends
                            
                            annotated_image = frame.copy()
                            inference_image = frame.copy()

                            detections = Inference(inference_image, 1)

                            smallest_difference = float('inf')
                            best_corners = None
                            best_center_x = None

                            for idx, det in enumerate(detections):
                                if len(det) == 0:
                                    continue
                                for thing in det:
                                    x_min, y_min, x_max, y_max, confidence = thing

                                    if class_names_evaczone[idx] != balls[ballIndex]:  # Ignore class with index 1
                                        continue
                                    
                                    if confidence > 0.2:  # Filter low-confidence detections
                                        image_width, image_height = frame_height, frame_width
                                        new_width, new_height = image_height, image_width
                                    
                                        x_new_min = y_min
                                        x_new_max = y_max
                                        y_new_min = new_width - x_max
                                        y_new_max = new_width - x_min
                                    
                                        y_new_min = new_height - y_new_min
                                        y_new_max = new_height - y_new_max
                                    
                                        x_min = int(x_new_min * image_width)
                                        y_min = int(y_new_min * image_height)
                                        x_max = int(x_new_max * image_width)
                                        y_max = int(y_new_max * image_height)

                                        corners = [
                                            (x_min, y_min),  # Top-left
                                            (x_max, y_min),  # Top-right
                                            (x_min, y_max),  # Bottom-left
                                            (x_max, y_max)   # Bottom-right
                                        ]
                                        center_x = (x_min + x_max) // 2
                                        center_y = (y_min + y_max) // 2

                                        # Calculate the difference between center_x and the middle of the frame
                                        difference_to_center = abs(center_x - (frame_width // 2))

                                        if difference_to_center < smallest_difference:
                                            smallest_difference = difference_to_center
                                            best_corners = corners
                                            best_center_x = center_x

                                        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                    
                                        label = f"{class_names_evaczone[idx]} {(confidence * 100):.0f}%"
                                        cv2.putText(annotated_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        cv2.putText(annotated_image, f"Diff: {difference_to_center}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                        cv2.putText(annotated_image, str(time.time() - start), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

                            if best_corners and best_center_x is not None:
                                # Calculate the distance between corner 0 and the bottom of the frame
                                corner_0 = best_corners[0]
                                distance_to_bottom = frame_height - corner_0[1]
                                
                                if distance_to_bottom < 10:
                                    runMotors(-60, 6)
                                    time.sleep(0.1)
                                    send_speciall_data(1)
                                    runMotors(-60, 6)
                                    time.sleep(1)
                                    runMotors(0, 0)
                                    time.sleep(0.5)

                                    if balls[ballIndex] == "Live victim":
                                        while True:
                                            ret, frame = get_picture(devMode, 1)
                                            annotated_image = frame.copy()
                                            # Keyframe green from the frame and get the biggest bounding box
                                            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                            lower_green = np.array([35, 100, 100])  # Adjust these values as needed
                                            upper_green = np.array([85, 255, 255])  # Adjust these values as needed
                                            green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

                                            # Find contours in the green mask
                                            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                            if green_contours:
                                                # Find the largest green contour by area
                                                largest_green_contour = max(green_contours, key=cv2.contourArea)
                                                x, y, w, h = cv2.boundingRect(largest_green_contour)
                                                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                displayGUI(None, None, annotated_image)
                                                if w > 0.05 * frame_width:
                                                    if w > 0.9 * frame_width:
                                                        runMotors(100, 10)
                                                        time.sleep(3)
                                                        send_speciall_data(2)
                                                        runMotors(0, 0)
                                                        time.sleep(1)
                                                        runMotors(100, -10)
                                                        time.sleep(5)
                                                        ballIndex += 1
                                                        break
                                                    driveTowardsPoint(x + w // 2, y + h // 2, frame_height, frame_width)
                                                else:
                                                    runMotors(0, 10)
                                            else:
                                                runMotors(0, 10)

                                    elif balls[ballIndex] == "Dead victim":
                                        while True:
                                            ret, frame = get_picture(devMode, 1)
                                            annotated_image = frame.copy()
                                            # Keyframe green from the frame and get the biggest bounding box
                                            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                            lower_red = np.array([100, 150, 100])
                                            upper_red = np.array([130, 255, 255])
                                            red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

                                            # Find contours in the green mask
                                            green_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                            if green_contours:
                                                # Find the largest green contour by area
                                                largest_green_contour = max(green_contours, key=cv2.contourArea)
                                                x, y, w, h = cv2.boundingRect(largest_green_contour)
                                                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                displayGUI(None, None, annotated_image)
                                                if w > 0.05 * frame_width:
                                                    if w > 0.9 * frame_width:
                                                        runMotors(100, 10)
                                                        time.sleep(3)
                                                        send_speciall_data(2)
                                                        runMotors(0, 0)
                                                        time.sleep(1)
                                                        runMotors(100, -10)
                                                        time.sleep(5)
                                                        ballIndex += 1
                                                        break
                                                    driveTowardsPoint(x + w // 2, y + h // 2, frame_height, frame_width)
                                                else:
                                                    runMotors(0, 10)
                                            else:
                                                runMotors(0, 10)
                                else:
                                    driveTowardsPoint(best_center_x, center_y, frame_height, frame_width)
                            else:
                                runMotors(0, 10)
                            displayGUI(None, None, annotated_image)
                            
                        



                        ##Find exit




                        
                        forwardRight, forward, forwardLeft, rearRight = get_distances_full(ser, send_speciall_data)
                        exitFound = False
                        while not exitFound:
                            forwardRight, forward, forwardLeft, rearRight = get_distances_full(ser, send_speciall_data)
                            desired_distance = 200  # Desired distance from the wall in mm
                            forwardRight = forwardRight * math.cos(math.radians(30)) + 10
                            difference = desired_distance - forwardRight
                            radius = (400 - abs(difference * 10)) * math_sign(difference)
                            runMotors(radius, 10)
                            print(f"Desired Angle: {difference}")
                            print("forward")
                    else:
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
                elif class_names[closest_class] == "T-intersection" or class_names[closest_class] == "X-intersection":  #T-Intersection
                    if class_names[closest_class] == "T-intersection":
                        print("T-Intersection")
                    else:
                        print("X-Intersection")
                    right = False
                    left = False

                    for idx, det in enumerate(detections):
                        if len(det) == 0:
                            continue
                        for idx2, thing in enumerate(det):
                            if class_names[idx] == "Green":
                                x_min, y_min, x_max, y_max, confidence = thing

                                if confidence > 0.6:  # Filter low-confidence detections
                                    new_width, new_height = image_height, image_width
                                
                                    x_new_min = y_min
                                    x_new_max = y_max
                                    y_new_min = new_width - x_max
                                    y_new_max = new_width - x_min
                                
                                    y_new_min = new_height - y_new_min
                                    y_new_max = new_height - y_new_max
                                
                                    x_min = int(x_new_min * image_width)
                                    y_min = int(y_new_min * image_height)
                                    x_max = int(x_new_max * image_width)
                                    y_max = int(y_new_max * image_height)
                                
                                    corners = [
                                        (x_min, y_min),  # Top-left
                                        (x_max, y_min),  # Top-right
                                        (x_min, y_max),  # Bottom-left
                                        (x_max, y_max)   # Bottom-right
                                    ]

                                    # Calculate the middle of the box
                                    box_middle_x = (x_min + x_max) / 2
                                    box_middle_y = (y_min + y_max) / 2

                                    # Calculate the middle of the closest corner
                                    intersection_box_middle_x = (closest_corners[0][0] + closest_corners[3][0]) / 2
                                    intersection_box_middle_y = (closest_corners[0][1] + closest_corners[3][1]) / 2

                                    if box_middle_y > intersection_box_middle_y:  # Below the X-Intersection
                                        if box_middle_x < intersection_box_middle_x:
                                            left = True
                                            print("Left")
                                        else:
                                            right = True
                                            print("Right")
                    if right and left:
                        print("U-TURN")
                        runMotors(0, 10)
                        pointTurn(devMode, get_picture, 180)
                    elif right:
                        print("Right")
                        runMotors(100, 10)
                        forwardDrive(devMode, get_picture, 0.6)
                        runMotors(0.1, 10)
                        pointTurn(devMode, get_picture, 90)
                    elif left:
                        print("Left")
                        runMotors(100, 10)
                        forwardDrive(devMode, get_picture, 0.6)
                        runMotors(-0.1, 10)
                        pointTurn(devMode, get_picture, 90)
                    else:
                        print("Forward")
                        runMotors(100, 10)
                        forwardDrive(devMode, get_picture, 0.6)
                    
                    annotated_image = None
                
    else:
        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
    
    if annotated_image is not None:
        elapsed_time = time.time() - start - inference_time
        cv2.putText(annotated_image, f"Inference Time: {inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Processing Time: {elapsed_time:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        displayGUI(annotated_image)

# Release resources
cv2.destroyAllWindows()
runMotors(0, 0)