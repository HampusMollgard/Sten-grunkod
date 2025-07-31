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
cv2.namedWindow("STEN X", cv2.WINDOW_NORMAL)
cv2.resizeWindow("STEN X", 640, 640)  # Set the size to 800x600

prevEntryPoint = None

min_distance = 100

class_names = ["90 degree", "Entry_point", "Green", "T-intersection", "X-intersection"]
class_names_evaczone = ["Dead victim", "Green", "Live victim", "Red", "Exit"]

if not devMode:
    from gpiozero import Button
    from picamera2 import Picamera2
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams,
                            FormatType)
    import serial

    button = Button(3)

    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

    picam2down = Picamera2(1)
    video_config = picam2down.create_video_configuration(
    main={"size": (2304, 1296), "format": "YUV420"},
    lores={"size": (1152, 648), "format": "YUV420"},
    display="lores",  # Optional: show preview from lores stream
    )

    picam2down.configure(video_config)  # Adjusted resolution without invalid 'controls' key
    # Uncomment the following line to set specific controls like exposure time
    # picam2down.set_controls({'ExposureTime': 1000})
    picam2down.start()


    picam2forward = Picamera2(0)
    video_config = picam2forward.create_video_configuration(
    main={"size": (2304, 1296), "format": "YUV420"},
    lores={"size": (1152, 648), "format": "YUV420"},
    display="lores",  # Optional: show preview from lores stream
    )

    picam2forward.configure(video_config)  # Adjusted resolution without invalid 'controls' key
    # Uncomment the following line to set specific controls like exposure time
    # picam2down.set_controls({'ExposureTime': 1000})
    picam2forward.start()

    output_filename = '/home/malte/film1.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    recordingFPS = 40
    recording_size = (640, 640)
    out = cv2.VideoWriter(output_filename, fourcc, recordingFPS, recording_size)

    target = VDevice()

    hef_path = '/home/malte/Downloads/best14.hef'
    hef = HEF(hef_path)

    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info.shape

    hef_path2 = '/home/malte/Downloads/balls4.hef'
    hef2 = HEF(hef_path2)

    configure_params2 = ConfigureParams.create_from_hef(hef=hef2, interface=HailoStreamInterface.PCIe)
    network_groups2 = target.configure(hef2, configure_params2)
    network_group2 = network_groups2[0]
    network_group_params2 = network_group2.create_params()

    input_vstreams_params2 = InputVStreamParams.make(network_group2, format_type=FormatType.UINT8)
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
    if img1_ is not None:
        current_time = time.time()
        if last_time is not None:
            fps = 1 / (current_time - last_time)
            last_time = current_time
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
    cv2.imshow('STEN X', combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        runMotors(0, 0)
        exit()

def Inference(frame_, model_index):
    if model_index == 0:
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):

                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                image_input = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
                input_data = {input_vstream_info.name: image_input}
                infer_results = infer_pipeline.infer(input_data)
                detections = infer_results['yolov8n/yolov8_nms_postprocess'][0] 
                return detections
    else:
        with InferVStreams(network_group2, input_vstreams_params2, output_vstreams_params2) as infer_pipeline2:
            with network_group2.activate(network_group_params2):

                frame_rgb = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
                image_input = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
                input_data = {input_vstream_info.name: image_input}
                infer_results = infer_pipeline2.infer(input_data)
                detections = infer_results['yolov8n/yolov8_nms_postprocess'][0] 
                return detections
    
def get_picture(dev_, camera_index):
    ret_ = False 
    if dev_:
        ret_, frame_ = cap.read()
    else:
        if camera_index == 0:
            frame__ = picam2down.capture_array("lores")
        else:
            frame__ = picam2forward.capture_array("lores")
        frame_ = cv2.cvtColor(frame__, cv2.COLOR_YUV2RGB_I420)
        frame_height, frame_width = frame_.shape[:2]
        square_size = int(0.86 * frame_height)
        x_start = (frame_width - square_size) // 2
        y_start = 0
        square_crop = frame_[y_start:y_start + square_size, x_start:x_start + square_size]
        frame_ = cv2.resize(square_crop, (640, 640), interpolation=cv2.INTER_AREA)
        ret_ = True
    if frame_.shape[-1] == 4:
        frame_ = frame_[:, :, :3]
    
    return ret_, frame_

def math_sign(value):
    return (value > 0) - (value < 0)

def send_data(val1, val2, val3):
    message = f"{val1},{val2},{val3}\n"  # Send as CSV with newline
    ser.write(message.encode())  
    print(f"Sent data: {message.strip()}")  # Print the sent message for debugging

def send_secret_data(val1, val2):
    message = f"{val1},{val2}\n"  # Send as CSV with newline
    ser.write(message.encode())  

def runMotors(radius, wanted_speed_outside):
    reversed = 1
    if radius > 0:
        reversed = 0
    if not devMode:
        send_data(round(abs(radius)), wanted_speed_outside, reversed)

def driveTowardsPoint(cX, cY, frame_height, frame_width, speed):
    x_point, y_point = cX, cY
    lower_middle_x = frame_width / 2

    # Calculate the differences
    dx = x_point - lower_middle_x
    dy = frame_height - y_point
    print(f"dx: {dx}, dy: {dy}")

    # Calculate the angle in degrees
    DrivingAngle = math.degrees(math.atan2(dy, dx))
    if DrivingAngle > 90:
        DrivingAngle = -180 + DrivingAngle
    
    #if cY > 400:
    kI = 0.05
    if not hasattr(driveTowardsPoint, "integral"):
        driveTowardsPoint.integral = 0
    if math_sign(DrivingAngle) == math_sign(driveTowardsPoint.integral):
        driveTowardsPoint.integral = 0
    driveTowardsPoint.integral += DrivingAngle
    if DrivingAngle > 0:
        DrivingAngle = abs(0.01*abs(DrivingAngle)+pow(1.08, abs(DrivingAngle) - 20)-abs(driveTowardsPoint.integral*kI)) #possible att stalla upp reaktiviteten
    else:
        DrivingAngle = -abs(0.01*abs(DrivingAngle)+pow(1.08, abs(DrivingAngle) - 20)-abs(driveTowardsPoint.integral*kI))

    runMotors(DrivingAngle, speed)

def normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width):
    
    if button.is_pressed:
        drive_round_object()
        return

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
        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 10)
    else:
        if top_most_point[1] < frame_height - 230:
            print("Gap detected, driving towards end")
            driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 10)
        else:
            print("Gap")
            contour_width = cv2.boundingRect(largest_contour)[2]

            if contour_width < 0.35 * frame.shape[1]:
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Draw the approximated corners on the annotated image
                if len(approx_corners) >= 4:
                    # Sort the points based on their y-coordinates
                    sorted_points = sorted(approx_corners, key=lambda point: point[0][1])

                    # Get the top two points and bottom two points
                    top_points = sorted_points[:2]
                    bottom_points = sorted_points[-2:]

                    # Calculate the average of the top points and bottom points
                    avg_top = np.mean([point[0] for point in top_points], axis=0).astype(int)
                    avg_bottom = np.mean([point[0] for point in bottom_points], axis=0).astype(int)

                    # Draw the average top and bottom points
                    cv2.circle(annotated_image, tuple(avg_top), 10, (0, 255, 255), -1)  # Yellow for avg_top
                    cv2.circle(annotated_image, tuple(avg_bottom), 10, (255, 255, 0), -1)  # Cyan for avg_bottom

                    displayGUI(None, annotated_image)

                    # Calculate the direction of the line
                    dx = avg_bottom[0] - avg_top[0]
                    dy = avg_bottom[1] - avg_top[1]
                    angle = math.degrees(math.atan2(dy, dx))
                    angle = -(math_sign(angle) * abs(angle - 90))
                    pointTurnIntersection(devMode, get_picture, angle)
                    print(f"Contour orientation angle: {angle:.2f} degrees")
                runMotors(-1000, 1)
                time.sleep(0.2)
                runMotors(-1000, 10)
                time.sleep(0.5)
                top_most_point = (top_most_point[0], 1000)
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
                    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        bounding_box_area = w * h
                        image_area = frame.shape[0] * frame.shape[1]

                        if bounding_box_area >= 0.05 * image_area:
                            top_most_point = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                            cv2.circle(annotated_image, top_most_point, 15, (255, 0, 0), -1)

                    displayGUI(annotated_image)
                print("DOne with gap")
            else:
                forwardDrive(devMode, get_picture, -15)
                time.sleep(1.5)
runMotors(0, 0)
time.sleep(1.5)
send_secret_data(3, 5)

def drive_round_object():
    runMotors(-700, -4)
    time.sleep(0.7)
    print("turning left")
    runMotors(-0.1, 10)
    pointTurnIntersection(devMode, get_picture, 90)
    runMotors(0, 0)
    time.sleep(1.5)
    runMotors(18, 10)
    while True:
        ret, frame = get_picture(devMode, 0)
        annotated_image = frame.copy()
        if not ret:
            print("End of video or cannot read frame.")
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Black mask
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)

        # Remove the upper part of the mask
        height, width = mask.shape
        upper_part_y_end = height // 2  # Define the upper half of the mask
        mask[:upper_part_y_end, :] = 0  # Set the upper part to 0
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(annotated_image, [largest_contour], -1, (0, 255, 0), 2)

            displayGUI(None, annotated_image)
            # Calculate the bounding box area of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if the contour is connected to the right side of the image
            connected_to_right = (x + w) >= frame.shape[1] - 1

            # Break the loop if the width of the rectangle exceeds 70% of the image width and is connected to the right side
            if connected_to_right and w > 0.7 * frame.shape[1]:
                print("Largest black contour width exceeds 70% of the image width and is connected to the right side. Exiting loop.")
                break
    
    runMotors(-700, 10)
    forwardDrive(devMode, get_picture, 0.5)
    runMotors(-0.1, 10)
    pointTurn(devMode, get_picture, 90)

def pointTurnIntersection(devMode, get_picture, angle):
    runMotors(0, 0)
    send_secret_data(1, angle)
    while True:
        ret, tempframe = get_picture(devMode, 0)
        if not ret:
            print("End of video or cannot read frame.")
            break
        displayGUI(tempframe)

        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            if response == "DONE":
                break
            else:
                print("Arduino says:", response)
    return
    

def pointTurn(devMode, get_picture, angle):
    start_time = time.time()
    while time.time() - start_time < 1.4 * (angle / 90):
        print("Running")
        ret, tempframe = get_picture(devMode, 0)
        annotated_image = tempframe.copy()

        hsv_frame = cv2.cvtColor(tempframe, cv2.COLOR_BGR2HSV)
    
        # Black mask
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)

        # Crop the mask to the lower middle part of the image
        height, width = mask.shape
        lower_middle_x_start = width // 3
        lower_middle_x_end = 2 * (width // 3)
        lower_middle_y_start =  5 * (height // 6)
        lower_middle_y_end = height

        cropped_mask = mask[lower_middle_y_start:lower_middle_y_end, lower_middle_x_start:lower_middle_x_end]

        # Find contours in the cropped mask
        contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]

        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate the moments of the largest contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                # Calculate the center of the largest black area
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(annotated_image, (cX + lower_middle_x_start, cY + lower_middle_y_start), 10, (255, 0, 255), -1)

                if angle > 0:
                    if cX >= cropped_mask.shape[1] // 2 and cY <= 2 * cropped_mask.shape[0] // 3:
                        print("Broke")
                        break
                else:
                    if cX <= cropped_mask.shape[1] // 2 and cY <= 2 * cropped_mask.shape[0] // 3:
                        print("broke")
                        break
            
        displayGUI(annotated_image)
    return

def forwardDrive(devMode, get_picture, drivingDistance):
    ser.flushInput()  # Clear any previous data in the serial buffer
    send_secret_data(2, drivingDistance)
    while True:
        ret, tempframe = get_picture(devMode, 0)
        if not ret:
            print("End of video or cannot read frame.")
            break
        displayGUI(tempframe)

        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            if response == "DONE":
                break
            else:
                print("Arduino says:", response)
    return

red_contour_timeout = time.time()
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

    lower_red2 = np.array([100, 140, 80])
    upper_red2 = np.array([130, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    # Find contours in the red mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if red_contours and time.time() - red_contour_timeout > 2:
        largest_red_contour = max(red_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_red_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if w >= 0.9 * frame.shape[1]:
            runMotors(0, 0)
            time.sleep(6)
            red_contour_timeout = time.time()

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
    closest_corner = None
    min_distance = float('inf')
    closest_index = -1  # Initialize closest index
    closest_corners = None

    for idx, det in enumerate(detections):
        if len(det) == 0:
            continue
        for idx2, thing in enumerate(det):
            #if len(det) > 1 and class_names[idx] == "90 degree" and idx2 < len(det) - 1:
            #    continue
            x_min, y_min, x_max, y_max, confidence = thing

            if confidence > 0.20:  # Filter low-confidence detections
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
                
                if class_names[idx] == "90 degree":
                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                else:
                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            
                label = f"{class_names[idx]} {(confidence * 100):.0f}%"
                cv2.putText(annotated_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(annotated_image, str(time.time() - start), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

                if class_names[idx] != "Green":  # Ignore class with index 1
                    objectDetected = True
                    for corner in corners:
                        distance = math.sqrt((corner[0] - bottom_most_point[0]) ** 2 + (corner[1] - bottom_most_point[1]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_corner = corner
                            closest_corners = corners
                            closest_index = (idx, idx2)






    #objectDetected = None






    if objectDetected:
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
            if (y_end - y_start) > 120:
                # Crop the mask
                cropped_mask = mask[y_start:y_end, x_start:x_end]
                if cropped_mask is not None:
                    mask_middle = np.mean(np.nonzero(cropped_mask), axis=1).astype(int)
                    cv2.circle(annotated_image, (x_start + mask_middle[1], y_start + mask_middle[0]), 10, (0, 255, 255), -1)
                    if (y_end - y_start) > 400:
                        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 10)
                    else:
                        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 6)
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

                    # Filter out contours smaller than 10 pixels
                    contours = [contour for contour in contours if cv2.contourArea(contour) > 20]

                    # Sort contours by area in descending order
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    # Get the coordinates of the two largest areas
                    if len(contours) == 2:
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
                        if magnitude1 != 0 and magnitude2 != 0:
                            angle = math.degrees(math.acos(dot_product / (magnitude1 * magnitude2)))

                            # Determine direction (right or left)
                            cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
                            direction = "right" if cross_product < 0 else "left"

                            angle = 180 - angle 

                            cv2.circle(annotated_image, closest_point, 15, (255, 0, 0), -1)

                            # Calculate the angle between a vertical line and the line from the bottom middle to center_x, center_y
                            vertical_vector = (0, -1)  # Vertical line pointing upwards
                            line_vector = (center_x - bottom_middle[0], center_y - (bottom_middle[1] + 100))
                            dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                            magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                            magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                            vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                            if center_x < frame_width // 2:
                                vertical_angle = -vertical_angle

                            # Calculate the length between bottom_middle and center_x / center_y
                            distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - (bottom_middle[1] + 100)) ** 2)

                            displayGUI(annotated_image, annotated_image)
                            print(f"Vertical Angle: {vertical_angle:.2f} degrees")

                            if (center_x - bottom_middle[0]) > 50:
                                pointTurnIntersection(devMode, get_picture, vertical_angle)
                            
                            forwardDrive(devMode, get_picture, 0 + (58 * (distance_to_center / 420)))

                            if cross_product < 0:
                                runMotors(0.1, 1)
                                time.sleep(0.3)
                                runMotors(0.1, 10)
                                pointTurn(devMode, get_picture, angle)
                            else:
                                runMotors(-0.1, 1)
                                time.sleep(0.3)
                                runMotors(-0.1, 10)
                                pointTurn(devMode, get_picture, angle)
                            
                            #runMotors(-700, -6)
                            #forwardDrive(devMode, get_picture, )
                            print("Turn  done")
                            runMotors(0, 0)


                            annotated_image = None
                    else:
                        print("Error when analysing the 90 degree object")
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
                elif class_names[closest_class] == "Entry_point":  # Green
                    x_start = closest_corners[0][0]
                    x_end = closest_corners[1][0]
                    object_width = abs(x_end - x_start)
                    displayGUI(None, annotated_image)

                    y_start = min(closest_corners[0][1], closest_corners[2][1])
                    y_end = max(closest_corners[0][1], closest_corners[2][1])
                    
                    object_middle_y = (y_start + y_end) // 2  # Calculate the middle Y-coordinate of the object
                    object_height = y_end - y_start  # Calculate the height of the object
                    
                    if object_width >= 0.9 * frame_width and object_height <= 0.4 * frame_height:  # Check width and height
                        send_secret_data(3, 6)
                        runMotors(-700, 10)
                        time.sleep(4)
                        runMotors(0, 0)
                        #balls = ["Exit"]
                        balls = ["Live victim", "Green", "Live victim", "Green", "Dead victim", "Red", "Exit"]
                        ballIndex = 0

                        start = time.time()
                        while ballIndex < len(balls):
                            if time.time() - start > 20 and balls[ballIndex] != "Exit":
                                ballIndex += 2
                                start = time.time()
                                continue
                            ret, frame = get_picture(devMode, 1)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            if not ret:
                                print("End of video or cannot read frame.")
                                break  # Exit loop when video ends
                            
                            annotated_image = frame.copy()
                            inference_image = frame.copy()

                            detections = Inference(inference_image, 1)

                            smallest_difference = float('inf')
                            best_corners = None
                            best_center_x = None

                            exit_corners = []
                            if balls[ballIndex] == "Exit":
                                for idx, det in enumerate(detections[class_names_evaczone.index(balls[ballIndex])]):
                                    print(len(detections[class_names_evaczone.index(balls[ballIndex])]))
                                    if len(detections[class_names_evaczone.index(balls[ballIndex])]) == 0:
                                        continue
                                    
                                    x_min, y_min, x_max, y_max, confidence = det
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
                                        exit_corners.append({
                                            "center_x": center_x,
                                            "center_y": center_y,
                                            "height": abs(corners[2][1] - corners[0][1])
                                        })
                                        best_corners = corners
                                        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                print("length")
                                print(len(exit_corners))
                                width_between_corners = 0
                                if len(exit_corners) > 1:
                                    # Find the two points closest to each other
                                    min_distance = float('inf')
                                    closest_pair = None
                                    for i, point1 in enumerate(exit_corners):
                                        for j, point2 in enumerate(exit_corners):
                                            if i != j:
                                                distance = math.sqrt((point1["center_x"] - point2["center_x"]) ** 2 +
                                                                     (point1["center_y"] - point2["center_y"]) ** 2)
                                                if distance < min_distance:
                                                    min_distance = distance
                                                    closest_pair = (point1, point2)
                                    
                                    best_center_x = ((closest_pair[0]["center_x"] + closest_pair[1]["center_x"]) // 2)
                                    # Calculate the width between the two closest corners
                                    width_between_corners = abs(closest_pair[0]["center_x"] - closest_pair[1]["center_x"])
                                    print(f"Width between corners: {width_between_corners}")
                                elif len(exit_corners) == 1:
                                    best_center_x = exit_corners[0]["center_x"]
                                    print(best_center_x)
                                

                            else:
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


                                            if class_names_evaczone[idx] != "Exit":
                                                difference_to_center = abs(center_x - (frame_width // 2))
                                            else:
                                                difference_to_center = abs(center_x - frame_width)

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
                                
                                if distance_to_bottom < 10 and balls[ballIndex] != "Exit":
                                    runMotors(-0.1, 10)
                                    pointTurnIntersection(devMode, get_picture, -7)
                                    send_secret_data(3, 1)
                                    time.sleep(1)
                                    runMotors(1000, 10)
                                    time.sleep(1)
                                    send_secret_data(3, 3)
                                    runMotors(0, 0)
                                    time.sleep(0.5)

                                    if balls[ballIndex] == "Live victim" or balls[ballIndex] == "Dead victim":
                                        ballIndex += 1
                                        print(balls[ballIndex])
                                        while True:
                                            ret, frame = get_picture(devMode, 1)
                                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                            if not ret:
                                                print("End of video or cannot read frame.")
                                                break  # Exit loop when video ends
                                            
                                            annotated_image = frame.copy()
                                            inference_image = frame.copy()

                                            detections = Inference(inference_image, 1)

                                            if len(detections[class_names_evaczone.index(balls[ballIndex])]) > 0:
                                                x_min, y_min, x_max, y_max, confidence = detections[class_names_evaczone.index(balls[ballIndex])][0]
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

                                                    w = x_max - x_min
                                                    h = y_max - y_min
                                                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                                    if w > 0.02 * frame_width:
                                                        if w > 0.9 * frame_width:
                                                            runMotors(-700, 10)
                                                            time.sleep(3)
                                                            send_secret_data(3, 2)
                                                            runMotors(0, 0)
                                                            time.sleep(1)
                                                            runMotors(-700, -10)
                                                            time.sleep(5)
                                                            runMotors(-0.1, 10)
                                                            pointTurnIntersection(devMode, get_picture, 180)
                                                            ballIndex += 1
                                                            start = time.time()
                                                            break
                                                        driveTowardsPoint(x_min + (w // 2.5), y_min + h // 2, frame_height, frame_width, 10)
                                                    else:
                                                        runMotors(0, 10)
                                                else:
                                                    runMotors(0, 10)
                                            else:
                                                runMotors(0, 10)

                                            displayGUI(None, None, annotated_image)
                                elif balls[ballIndex] == "Exit" and (distance_to_bottom < 190 or width_between_corners > 0.8 * frame_width):
                                    runMotors(-700, 10)

                                    exitFound = True
                                    timeOut = time.time()
                                    while time.time() - timeOut < 3:
                                        ret, frame = get_picture(devMode, 0)
                                        frame = cv2.resize(frame, (640, 640))
                                        annotated_image = frame.copy()
                                        inference_image = frame.copy()

                                        detections = Inference(inference_image, 0)

                                        for idx, det in enumerate(detections):
                                            for idx2, thing in enumerate(det):
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

                                                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                                    if class_names[idx] == "T-intersection" or class_names[idx] == "X-intersection":
                                                        runMotors(0, 0)
                                                        break
                                                    if class_names[idx] == "Entry_point":
                                                        exitFound = False
                                                        break

                                                    label = f"{class_names[idx]} {(confidence * 100):.0f}%"
                                                    cv2.putText(annotated_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                                    cv2.putText(annotated_image, str(time.time() - start), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                            
                                        displayGUI(annotated_image)
                                    
                                    if not exitFound:
                                        runMotors(-700, -10)
                                        time.sleep(4)
                                        runMotors(-0.1, 10)
                                        pointTurnIntersection(devMode, get_picture, 90)
                                    else:
                                        runMotors(-700, 10)
                                        time.sleep(1)
                                        send_secret_data(3, 5)
                                        break
                                else:
                                    driveTowardsPoint(best_center_x, center_y, frame_height, frame_width, 10)
                            else:
                                runMotors(0, 5)
                            displayGUI(None, None, annotated_image)
        

                    else:
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
                elif class_names[closest_class] == "T-intersection" or class_names[closest_class] == "X-intersection":  #T-Intersection
                    TintersectionOkay = True
                    if class_names[closest_class] == "T-intersection":
                        print("T-Intersection")
                        # Calculate the size of the T-intersection
                        intersection_width = abs(closest_corners[1][0] - closest_corners[0][0])
                        intersection_height = abs(closest_corners[2][1] - closest_corners[0][1])
                        print(f"T-Intersection Width: {intersection_width}, Height: {intersection_height}")
                        if 300 <= intersection_width <= 400 and 300 <= intersection_height <= 400:
                            print("T-Intersection size is within the range of 300-400.")
                        else:
                            TintersectionOkay = False

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
                    if TintersectionOkay:
                        if right and left:
                            print("U-TURN")
                            pointTurnIntersection(devMode, get_picture, -180)
                        elif right:
                            print("Right")
                            '''
                            forwardDrive(devMode, get_picture, 35)
                            pointTurnIntersection(devMode, get_picture, 90)
                            '''
                            runMotors(9, 10)
                            time.sleep(2)
                            runMotors(0, 0)
                        elif left:
                            print("Left")
                            '''
                            forwardDrive(devMode, get_picture, 35)
                            pointTurnIntersection(devMode, get_picture, -90)
                            '''
                            runMotors(-9, 10)
                            time.sleep(2)
                            runMotors(0, 0)
                        else:
                            print("Forward")
                            runMotors(-700, 10)
                            forwardDrive(devMode, get_picture, 50)
                    else:
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
                    
                    annotated_image = None
                
    else:
        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, largest_contour, bottom_most_point, frame_height, frame_width)
    
    if annotated_image is not None:
        totalTime = time.time() - start 
        elapsed_time = totalTime - inference_time
        
        cv2.putText(annotated_image, f"Inference Time: {inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Processing Time: {elapsed_time:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Totalt Time: {totalTime:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        displayGUI(annotated_image)