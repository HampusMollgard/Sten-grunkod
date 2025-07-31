import cv2
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import numpy as np
import requests
import supervision as sv
import math
import time
import struct
import serial
from gpiozero import Button
from picamera2 import Picamera2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                        ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams,
                        FormatType)


# Create OpenCV window for displaying the results
cv2.namedWindow("STEN 11", cv2.WINDOW_NORMAL)
cv2.resizeWindow("STEN 11", 640, 640)  # Set the size to 800x600

min_distance = 100

class_names = ["Entry_point", "Green", "T-intersection", "X-intersection", "None", "90 degree"]
#class_names = ["Entry_point", "Green", "T-intersection", "X-intersection"]
class_names_evaczone = ["Dead victim", "Green", "Live victim", "Red", "Exit"]

button = Button(3)
buttonL = Button(10)
buttonR = Button(9)

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

picam2down = Picamera2(1)
video_config = picam2down.create_video_configuration(
main={"size": (2304, 1296), "format": "YUV420"},
lores={"size": (1152, 648), "format": "YUV420"},
display="lores",
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
recordingFPS = 20
recording_size = (1280, 1280)
out = cv2.VideoWriter(output_filename, fourcc, recordingFPS, recording_size)

target = VDevice()

hef_path = '/home/malte/Downloads/best25.hef'
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

hef_path2 = '/home/malte/Downloads/best18.hef'
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

    # Save image into file
    #out.write(cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    cv2.imshow('STEN 11', combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        out.release()
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
    
def get_picture(camera_index):
    ret_ = False 
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
    print(f"Sent data: {message.strip()}")

def runMotors(radius, wanted_speed_outside):
    reversed = 1
    if radius > 0:
        reversed = 0
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
    kI = 0.00
    if not hasattr(driveTowardsPoint, "integral"):
        driveTowardsPoint.integral = 0
    if math_sign(DrivingAngle) == math_sign(driveTowardsPoint.integral):
        driveTowardsPoint.integral = 0
    driveTowardsPoint.integral += abs(DrivingAngle)

    kD = 0.3
    if not hasattr(driveTowardsPoint, "prevDrivingAngle"):
        driveTowardsPoint.prevDrivingAngle = DrivingAngle
    derivative = DrivingAngle - driveTowardsPoint.prevDrivingAngle
    driveTowardsPoint.prevDrivingAngle = DrivingAngle
    if DrivingAngle > 0:
        DrivingAngle = abs(0.01*abs(DrivingAngle)+pow(1.12, abs(DrivingAngle) - 50)-abs(driveTowardsPoint.integral*kI)-(derivative*kD))
    else:
        DrivingAngle = -abs(0.01*abs(DrivingAngle)+pow(1.12, abs(DrivingAngle) - 50)+abs(driveTowardsPoint.integral*kI)+(derivative*kD))

    runMotors(DrivingAngle, speed)


def driveTowardsBall(cX, cY, frame_height, frame_width, speed): 
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
        DrivingAngle = abs(0.01*abs(DrivingAngle)+pow(1.08, abs(DrivingAngle) - 10)) #possible att stalla upp reaktiviteten
    else:
        DrivingAngle = -abs(0.01*abs(DrivingAngle)+pow(1.08, abs(DrivingAngle) - 10))

    runMotors(DrivingAngle, speed)

def normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, frame_height, frame_width):
    # Crop the mask to 50% height and 70% width, located at the lower middle edge
    mask_height, mask_width = mask.shape
    crop_height = int(mask_height * 1)
    crop_width = int(mask_width * 1)
    x_start = (mask_width - crop_width) // 2
    y_start = mask_height - crop_height
    cropped_mask = mask[y_start:y_start + crop_height, x_start:x_start + crop_width]

    # Find the top-most pixel location in the cropped mask
    nonzero_points = np.argwhere(cropped_mask > 0)

    if nonzero_points.size == 0:
        # Default to the middle if there are no points
        mask_middle = np.array([crop_height // 2, crop_width // 2])
        average_point = (x_start + mask_middle[1], y_start + mask_middle[0])
        top_most_point = (x_start + mask_middle[1], y_start + mask_middle[0])
        leftmost_point = (x_start, y_start + mask_middle[0])
        rightmost_point = (x_start + crop_width - 1, y_start + mask_middle[0])
    else:
        # Top-most pixel (minimum y)
        top_idx = np.argmin(nonzero_points[:, 0])
        top_most_pixel = nonzero_points[top_idx]
        top_most_point = (x_start + top_most_pixel[1], y_start + top_most_pixel[0])

        # Average point
        mask_middle = np.mean(nonzero_points, axis=0).astype(int)
        average_point = (x_start + mask_middle[1], y_start + mask_middle[0])

        # Left-most pixel (minimum x)
        left_idx = np.argmin(nonzero_points[:, 1])
        leftmost_pixel = nonzero_points[left_idx]
        leftmost_point = (x_start + leftmost_pixel[1], y_start + leftmost_pixel[0])

        # Right-most pixel (maximum x)
        right_idx = np.argmax(nonzero_points[:, 1])
        rightmost_pixel = nonzero_points[right_idx]
        rightmost_point = (x_start + rightmost_pixel[1], y_start + rightmost_pixel[0])

    pointToFollow = average_point
    

    '''
    # If the top-most point is in the lower part of the image, check which side deviates most from the middle
    if top_most_point[1] > frame_height * 0.3:
        middle_x = frame_width // 2
        left_deviation = abs(leftmost_point[0] - middle_x)
        right_deviation = abs(rightmost_point[0] - middle_x)
        if left_deviation > right_deviation:
            pointToFollow = leftmost_point
        else:
            pointToFollow = rightmost_point
            '''
    

    cv2.circle(annotated_image, top_most_point, 10, (255, 0, 0), -1)
    cv2.circle(annotated_image, (x_start + mask_middle[1], y_start + mask_middle[0]), 10, (0, 255, 255), -1)
    cv2.circle(annotated_image, pointToFollow, 30, (255, 255, 255), -1)
    displayGUI(annotated_image, cropped_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_width = cv2.boundingRect(largest_contour)[2]
    else:
        largest_contour = None
        contour_width = 0

    if top_most_point[1] < frame_height - 170:
        driveTowardsPoint(pointToFollow[0], pointToFollow[1], frame_height, frame_width, 10)
    else:
        print("Gap")
        runMotors(0, 0)

        ret, frame = get_picture(0)
        if not ret:
            print("End of video or cannot read frame.")
            return
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        mask = cv2.inRange(hsv_frame, lower_black, upper_black)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            largest_contour = None
        contour_width = cv2.boundingRect(largest_contour)[2]

        if contour_width < 0.35 * frame.shape[1]:
            try:
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
                    angle = abs(angle - 90)
                    if dx > 0:
                        angle = -angle
                    print(f"Contour orientation angle: {angle:.2f} degrees")
                    pointTurnIntersection(get_picture, angle)
                    print(f"Contour orientation angle: {angle:.2f} degrees")
                runMotors(1000, 1)
                time.sleep(0.2)
                runMotors(10000, 10)
                time.sleep(0.5)
                top_most_point = (top_most_point[0], 1000)
                while top_most_point[1] > frame_height - 170:
                    ret, frame = get_picture(0)
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
            except:
                pass
        else:
            forwardDrive(get_picture, -15)
            time.sleep(1.5)
runMotors(0, 0)
time.sleep(0.3)
send_secret_data(3, 5)
time.sleep(2)

def pointTurnIntersection(get_picture, angle):
    runMotors(0, 0)
    send_secret_data(1, angle)
    while True:
        print("Point turning")
        ret, tempframe = get_picture(0)
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

def pointTurn(get_picture, angle):
    runMotors(0, 0)
    send_secret_data(1, angle)
    while True:
        print("Point turning")
        ret, tempframe = get_picture(0)
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

def forwardDrive(get_picture, drivingDistance):
    print("Driving forward:", drivingDistance)
    ser.flushInput()  # Clear any previous data in the serial buffer
    send_secret_data(2, drivingDistance)
    while True:
        ret, tempframe = get_picture(0)
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
exited = False
hasExited = False
red_contour_timeout = time.time()
def balls(class_names_evaczone, button, buttonL, buttonR, ser, displayGUI, Inference, get_picture, send_secret_data, runMotors, driveTowardsBall, pointTurnIntersection, forwardDrive, frame_height, frame_width):
    send_secret_data(3, 6)
    runMotors(-70, 10)
    time.sleep(2)
    runMotors(0, 0)
    #balls = ["Exit"]
    #balls = ["Live victim", "Green", "Live victim", "Green", "Dead victim", "Red", "Exit"]
    balls = ["Live victim", "Exit"]
    ballIndex = 0

    while ballIndex == 0:
        ret, frame = get_picture(1)
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
        start = time.time()
        for idx, det in enumerate(detections):
            if len(det) == 0:
                continue
            for thing in det:
                x_min, y_min, x_max, y_max, confidence = thing

                if class_names_evaczone[idx] != balls[ballIndex]:  # Ignore class with index 1
                    continue
                                    
                if confidence > 0.1:  # Filter low-confidence detections
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
        displayGUI(None, None , annotated_image)

        if best_corners and best_center_x is not None:
                                # Calculate the distance between corner 0 and the bottom of the frame
            corner_0 = best_corners[0]
            distance_to_bottom = frame_height - corner_0[1]
                                
            if distance_to_bottom < 10 and balls[ballIndex] != "Exit":
                pointTurnIntersection(get_picture, -7)
                send_secret_data(3, 1)
                time.sleep(1)
                runMotors(1000, 10)
                time.sleep(2)
                runMotors(0, 0)
                time.sleep(0.3)
                send_secret_data(3, 3)
                                    
                pickedBall = True
                start_time = time.time()

                while True:
                    if time.time() - start_time > 5:
                        print("Timeout reached")
                        break

                    if ser.in_waiting > 0:
                        response = ser.readline().decode().strip()
                        print(response)
                        if response == "DONE":
                            ballIndex = ballIndex + 1
                            send_secret_data(3, 5)
                            break
                        elif response == "FAILED":
                            forwardDrive(get_picture, -100)
                            break
                        else:
                            print(response)
            else:
                driveTowardsBall(best_center_x, center_y, frame_height, frame_width, 10)
        else:
            runMotors(0, 5)

    temp = True
    while temp:
        runMotors(-0.1, 8)
        ret, frame = get_picture(1)
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
        start = time.time()
        for idx, det in enumerate(detections):
            if len(det) == 0:
                continue
            for thing in det:
                x_min, y_min, x_max, y_max, confidence = thing

                if class_names_evaczone[idx] != "Exit":  # Ignore class with index 1
                    continue
                                    
                if confidence > 0.2:  # Filter low-confidence detections
                    pointTurnIntersection(get_picture, 45)
                    temp = False

    while True:
        runMotors(-1000, 20)
        if button.is_pressed or buttonL.is_pressed:
            forwardDrive(get_picture, -50)
            pointTurnIntersection(get_picture, -30)
            break
        if buttonR.is_pressed:
            break
        time.sleep(0.1)

    exited = False
    while not exited:
        ret, frame = get_picture(0)
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
        fountEntry = False

        if not fountEntry:
                                # Mask out black if the size exceeds 10,000 px
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 80])
            mask_black = cv2.inRange(hsv_frame, lower_black, upper_black)
            contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_black:
                if cv2.contourArea(contour) > 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 0.8 * frame.shape[1] and h > 0.2 * frame.shape[0]:
                        print("exit")
                        cv2.drawContours(annotated_image, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

                        forwardDrive(get_picture, 100)
                        time.sleep(1)
                        send_secret_data(3, 5)
                        exited = True
        displayGUI(None, annotated_image)

                            #TOGGLE EVAC
        if button.is_pressed or buttonL.is_pressed:
            forwardDrive(get_picture, -50)
            pointTurnIntersection(get_picture, -60)
        elif buttonR.is_pressed:
            pointTurnIntersection(get_picture, -20)
        else:
            runMotors(40, 18)

        time.sleep(0.1)
    return image_height,image_width,start,frame,annotated_image,detections, exited

while True:
    if exited and buttonR.is_pressed and buttonL.is_pressed:
        send_secret_data(3, 2)
        forwardDrive(get_picture, -100)
        pointTurnIntersection(get_picture, -180)
        exited = False
    
    if exited and (buttonR.is_pressed or buttonL.is_pressed):
        runMotors(1000, 0)
        time.sleep(0.3)

    start = time.time()
    ret, frame = get_picture(0)
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
    # Remove red mask from black mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(red_mask))
    
    # Find contours in the red mask
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if red_contours and time.time() - red_contour_timeout > 2:
        largest_red_contour = max(red_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_red_contour)
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if w >= 0.9 * frame.shape[1] and h >= 0.5 *frame.shape[0]:
            runMotors(0, 0)
            time.sleep(6)
            red_contour_timeout = time.time()

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
    
    bottom_most_point = (320, 640)
    closest_class = 5

    for idx, det in enumerate(detections):
        if len(det) == 0:
            continue
        for idx2, thing in enumerate(det):
            #if len(det) > 1 and class_names[idx] == "90 degree" and idx2 < len(det) - 1:
            #    continue
            x_min, y_min, x_max, y_max, confidence = thing

            if (confidence > 1 and class_names[idx] == "90 degree") or confidence > 0.20 and class_names[idx] != "90 degree":  # Filter low-confidence detections
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

                if class_names[idx] != "Green" and class_names[idx] != "90 degree":  # Ignore class with index 1
                    # For intersections, check if all corners are at least 50 px from the edges
                    if class_names[idx] == "T-intersection" or class_names[idx] == "X-intersection":
                        corners_okay = all(
                            25 <= corner[0] <= (frame_width - 25) and
                            0 <= corner[1] <= (frame_height - 0)
                            for corner in corners
                        )
                        if not corners_okay:
                            continue  # Skip this detection if any corner is too close to the edge
                    objectDetected = True
                    for corner in corners:
                        distance = math.sqrt((corner[0] - bottom_most_point[0]) ** 2 + (corner[1] - bottom_most_point[1]) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_corner = corner
                            closest_corners = corners
                            closest_index = (idx, idx2)

    #objectDetected = False

    if objectDetected:
        if closest_corner:
            cv2.circle(annotated_image, (int(closest_corner[0]), int(closest_corner[1])), 10, (255, 255, 0), -1)
            # Calculate vertical distance
            vertical_distance = abs(closest_corner[1] - bottom_most_point[1])

            # Calculate the width to maintain the aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            width = int(vertical_distance * aspect_ratio)

            # Calculate the cropping coordinates
            x_start = int(0)
            x_end = int(640)
            y_start = int(closest_corner[1])
            y_end = int(640)
            
            # Only crop the mask if the height is greater than 80
            print(class_names[closest_class])
            if (y_end - y_start) > 120 and (class_names[closest_class] != "Entry_point"):
                # Crop the mask
                cropped_mask = mask[y_start:y_end, x_start:x_end]
                displayGUI(None, cropped_mask)
                if cropped_mask is not None:
                    mask_middle = np.mean(np.nonzero(cropped_mask), axis=1).astype(int)
                    cv2.circle(annotated_image, (x_start + mask_middle[1], y_start + mask_middle[0]), 10, (0, 255, 255), -1)
                    if (y_end - y_start) > 400:
                        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 10)
                    else:
                        driveTowardsPoint(x_start + mask_middle[1], y_start + mask_middle[0], frame_height, frame_width, 10)
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
                    contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

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

                            cv2.circle(annotated_image, closest_point, 30, (255, 0, 0), -1)

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

                            if (center_x - bottom_middle[0]) > 0:
                                pointTurnIntersection(get_picture, vertical_angle)
                            
                            forwardDrive(get_picture, 0 + (58 * (distance_to_center / 420)))

                            if cross_product < 0:
                                pointTurn(get_picture, angle)
                            else:
                                pointTurn(get_picture, -angle)
                            
                            print("Turn  done")
                            runMotors(0, 0)


                            annotated_image = None
                    else:
                        print("Error when analysing the 90 degree object")
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, frame_height, frame_width)
                elif class_names[closest_class] == "Entry_point" and not exited and not hasExited:  # Green
                    x_start = closest_corners[0][0]
                    x_end = closest_corners[1][0]
                    object_width = abs(x_end - x_start)
                    displayGUI(None, annotated_image)

                    y_start = min(closest_corners[0][1], closest_corners[2][1])
                    y_end = max(closest_corners[0][1], closest_corners[2][1])
                    
                    object_middle_y = (y_start + y_end) // 2  # Calculate the middle Y-coordinate of the object
                    object_height = y_end - y_start  # Calculate the height of the object
                    
                    if object_width >= 0.7 * frame_width and object_height <= 0.7 * frame_height:  # Check width and height
                        image_height, image_width, start, frame, annotated_image, detections, exited = balls(class_names_evaczone, button, buttonL, buttonR, ser, displayGUI, Inference, get_picture, send_secret_data, runMotors, driveTowardsBall, pointTurnIntersection, forwardDrive, frame_height, frame_width)
                        hasExited = True           

                    else:
                        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, frame_height, frame_width)
                elif class_names[closest_class] == "T-intersection" or class_names[closest_class] == "X-intersection":  #T-Intersection
                    if class_names[closest_class] == "T-intersection":
                        print("T-Intersection")
                    else:
                        print("X-Intersection")
                    # Check if intersection corners are at least 50 px from the edge
                    TintersectionOkay = True
                    for corner in closest_corners:
                        if corner[0] < 25 or corner[0] > (frame_width - 25):
                            TintersectionOkay = False
                            break
                    
                    right = False
                    left = False
                    intersection_box_middle_x = None

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
                    
                    x_start = closest_corners[0][0]
                    x_end = closest_corners[1][0]
                    y_start = min(closest_corners[0][1], closest_corners[2][1])
                    y_end = max(closest_corners[0][1], closest_corners[2][1])

                    # Calculate the center point
                    center_x = (x_start + x_end) // 2
                    center_y = (y_start + y_end) // 2
                    if TintersectionOkay:
                        if right and left:
                            print("U-TURN")

                            bottom_middle = (frame_width // 2, frame_height - 1)


                            vertical_vector = (0, -1)  # Vertical line pointing upwards
                            line_vector = (center_x - bottom_middle[0], center_y - (bottom_middle[1] + 100))
                            dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                            magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                            magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                            vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                            distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - (bottom_middle[1] + 100)) ** 2)

                            if (center_x - bottom_middle[0]) > 50:
                                pointTurnIntersection(get_picture, vertical_angle)
                            
                            forwardDrive(get_picture, 0 + (58 * (distance_to_center / 420)))
                                
                            forwardDrive(get_picture, 20)
                        elif right:
                            print("Right")

                            bottom_middle = (frame_width // 2, frame_height - 1)

                            vertical_vector = (0, -1)  # Vertical line pointing upwards
                            line_vector = (center_x - bottom_middle[0], center_y - (bottom_middle[1] + 100))
                            dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                            magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                            magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                            vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                            distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - (bottom_middle[1] + 100)) ** 2)

                            if (center_x - bottom_middle[0]) > 50:
                                pointTurnIntersection(get_picture, vertical_angle)
                            
                            forwardDrive(get_picture, 0 + (58 * (distance_to_center / 420)))
                            
                            pointTurnIntersection(get_picture, 90)
                            '''
                            runMotors(9, 10)
                            time.sleep(2)
                            '''
                            runMotors(0, 0)
                        elif left:
                            print("Left")
                            
                            bottom_middle = (frame_width // 2, frame_height - 1)

                            vertical_vector = (0, -1)  # Vertical line pointing upwards
                            line_vector = (center_x - bottom_middle[0], center_y - (bottom_middle[1] + 100))
                            dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                            magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                            magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                            vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                            distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - (bottom_middle[1] + 100)) ** 2)

                            if (center_x - bottom_middle[0]) > 50:
                                pointTurnIntersection(get_picture, vertical_angle)
                            
                            forwardDrive(get_picture, 0 + (58 * (distance_to_center / 420)))

                            pointTurnIntersection(get_picture, -90)

                            
                            '''
                            runMotors(-9, 10)
                            time.sleep(2)
                            '''
                            runMotors(0, 0)
                        else:
                            print("Forward")

                            bottom_middle = (frame_width // 2, frame_height - 1)

                            vertical_vector = (0, -1)  # Vertical line pointing upwards
                            line_vector = (center_x - bottom_middle[0], center_y - (bottom_middle[1] + 100))
                            dot_product_vertical = vertical_vector[0] * line_vector[0] + vertical_vector[1] * line_vector[1]
                            magnitude_vertical = math.sqrt(vertical_vector[0] ** 2 + vertical_vector[1] ** 2)
                            magnitude_line = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)
                            vertical_angle = math.degrees(math.acos(dot_product_vertical / (magnitude_vertical * magnitude_line)))

                            distance_to_center = math.sqrt((center_x - bottom_middle[0]) ** 2 + (center_y - (bottom_middle[1] + 100)) ** 2)

                            if (center_x - bottom_middle[0]) > 50:
                                pointTurnIntersection(get_picture, vertical_angle)
                            
                            forwardDrive(get_picture, 0 + (58 * (distance_to_center / 420)))
                                
                            forwardDrive(get_picture, 20)

                            ret, frame = get_picture(0)
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
                            fountEntry = False

                            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            lower_black = np.array([0, 0, 0])
                            upper_black = np.array([180, 255, 80])
                            mask_black = cv2.inRange(hsv_frame, lower_black, upper_black)
                            contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            contour_areas = [cv2.contourArea(contour) for contour in contours_black]
                            if contour_areas:
                                max_area = max(contour_areas)
                                largest_contour = contours_black[contour_areas.index(max_area)]
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                if max_area < 50000:
                                    runMotors(0, 0)
                                    time.sleep(10)
                                    # Gather necessary data for the balls function
                                    frame_height = frame.shape[0]
                                    frame_width = frame.shape[1]
                                    image_height, image_width, start, frame, annotated_image, detections, exited = balls(class_names_evaczone, button, buttonL, buttonR, ser, displayGUI, Inference, get_picture, send_secret_data, runMotors, driveTowardsBall, pointTurnIntersection, forwardDrive, frame_height, frame_width)


                                            
                        annotated_image = None
                
    else:
        normal_driving(runMotors, driveTowardsPoint, frame, annotated_image, mask, frame_height, frame_width)
    
    if annotated_image is not None:
        totalTime = time.time() - start 
        elapsed_time = totalTime - inference_time
        
        cv2.putText(annotated_image, f"Inference Time: {inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Processing Time: {elapsed_time:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Totalt Time: {totalTime:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        displayGUI(annotated_image)