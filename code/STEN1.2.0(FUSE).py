import cv2
import numpy as np
from math import cos, sin, radians
import math
import time
import matplotlib.pyplot as plt
import atexit

# Pin configuration
PWM_PINL = 13# Left PWM pin
PWM_PINR = 12#Right PWM pin
MotorLPin1 = 5
MotorLPin2 = 6
LedPin = 0
MotorL1 = 0
MotorL2 = 1#High = forward

MotorRPin1 = 16
MotorRPin2 = 26
MotorR1 = 0
MotorR2 = 1#High = forward

PWM_FREQUENCY = 1000  # Frequency in Hz (e.g., 1000 Hz)
devMode = True

# Open the video
if devMode:
    cap = cv2.VideoCapture('LiveRecording.mp4')
#cap = cv2.imread('image.png')

# Create a named window
cv2.namedWindow('Lines Detection', cv2.WINDOW_NORMAL)

# Move the window to the desired position (x, y)
window_x = 0  # Horizontal position
window_y = -50  # Vertical position
cv2.moveWindow('Lines Detection', window_x, window_y)
angle = 0
fit = None
errors = 0
lastX = 500
fps = 0
frame_count = 0
fps_start_time = time.time()

output_filename = '/home/malte/film1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recordingFPS = 50
recording_size = (640, 480)
out = cv2.VideoWriter(output_filename, fourcc, recordingFPS, recording_size)

def cleanupAtEnd():
    if devMode:
        cap.release()
        
    out.release()
    cv2.destroyAllWindows()

    if not devMode:
        runMotors(0, 0)
        time.sleep(1)
        runMotors(0, 0)
        lgpio.gpio_write(h, LedPin, 0)
        lgpio.gpiochip_close(h)
        print("GPIO closed.")

    print('program ended')

atexit.register(cleanupAtEnd)

if not devMode:
    from picamera2 import Picamera2
    import lgpio

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    #picam2.set_controls({'ExposureTime': 10000})
    picam2.start()

    # Open a handle to the GPIO chip
    h = lgpio.gpiochip_open(0)  # '0' usually corresponds to the main GPIO chip

    # Init Left side
    lgpio.gpio_claim_output(h, PWM_PINL)
    lgpio.gpio_claim_output(h, MotorLPin1)
    lgpio.gpio_write(h, MotorLPin1, MotorL1)
    lgpio.gpio_claim_output(h, MotorLPin2)
    lgpio.gpio_write(h, MotorLPin2, MotorL2)
    #init Right side
    lgpio.gpio_claim_output(h, PWM_PINR)
    lgpio.gpio_claim_output(h, MotorRPin1)
    lgpio.gpio_write(h, MotorRPin1, MotorR1)
    lgpio.gpio_claim_output(h, MotorRPin2)
    lgpio.gpio_write(h, MotorRPin2, MotorR2)
    #init Led
    lgpio.gpio_claim_output(h, LedPin)

def math_sign(value):
    return (value > 0) - (value < 0)

def runMotors(radius, wanted_speed_outside):
    # Start PWM on the pin
    #lgpio.tx_pwm(h, PWM_PINR, PWM_FREQUENCY, 100)
    #lgpio.tx_pwm(h, PWM_PINL, PWM_FREQUENCY, 100)  # 50% duty cycle
    
    print(radius)
    
    reversed = False
    if radius > 0:
        reversed = True
        
    radius = abs(radius)

    global current_speed_l, current_speed_r

    wheel_base_width = 20  # Example value, replace with the actual value
    wheel_base_length = 15  # Example value, replace with the actual value
    wanted_speed_outside = 100
    if reversed:
        outer_radius = radius + (wheel_base_width / 2)
        # current_speed_r = wanted_speed_outside
        current_speed_r = wanted_speed_outside
        degrees_per_second = wanted_speed_outside / (outer_radius * 2 * math.pi) * 360
        inner_radius = radius - (wheel_base_width / 2)
        current_speed_l = degrees_per_second * ((inner_radius * 2 * math.pi) / 360)
    else:
        outer_radius = radius + (wheel_base_width / 2)
        # current_speed_l = wanted_speed_outside
        current_speed_l = wanted_speed_outside
        degrees_per_second = wanted_speed_outside / (outer_radius * 2 * math.pi) * 360
        inner_radius = radius - (wheel_base_width / 2)
        current_speed_r = degrees_per_second * ((inner_radius * 2 * math.pi) / 360)
        
    #Skicka datan till motorerna
    print(current_speed_l, current_speed_r)
    
    if current_speed_l < 0:
        MotorL1 = 1
        MotorL2 = 0
    else:
        MotorL1 = 0
        MotorL2 = 1
    lgpio.gpio_write(h, MotorLPin1, MotorL1)
    lgpio.gpio_write(h, MotorLPin2, MotorL2)
    if current_speed_r < 0:
        MotorR1 = 1
        MotorR2 = 0
    else:
        MotorR1 = 0
        MotorR2 = 1
    lgpio.gpio_write(h, MotorRPin1, MotorR1)
    lgpio.gpio_write(h, MotorRPin2, MotorR2)
    lgpio.tx_pwm(h, PWM_PINR, PWM_FREQUENCY, abs(current_speed_r))
    lgpio.tx_pwm(h, PWM_PINL, PWM_FREQUENCY, abs(current_speed_l))
    lgpio.gpio_write(h, LedPin, 1)

def get_intersection_data(image, mask, x, y, angle, forward_distance):
    nIntersectedLines = []
    threashold = 1
    new_x_vertical = None
    new_y_vertical = None
    new_x_horizontal = None
    new_y_horizontal = None
    #cv2.circle(image, (int(x + (np.cos(np.radians(0) - angle) * radius)), int(y + (np.sin(np.radians(0) - angle) * radius))), 20, (255, 0, 0), -1)
    
    temp, width, pos, point = get_perpendicular_line(image, mask, x, y, angle, forward_distance)
    
    if width is not None and width > threashold:
        nIntersectedLines.append('up')
        new_x_vertical = point[0]
        new_y_vertical = point[1]
    
    temp, width, pos, point = get_perpendicular_line(image, mask, x, y, (angle - (np.pi / 2)), forward_distance)

    if width is not None and width > threashold:
        nIntersectedLines.append('left')
        new_x_horizontal = point[0]
        new_y_horizontal = point[1]

    temp, width, pos, point = get_perpendicular_line(image, mask, x, y, (angle + (np.pi / 2)), forward_distance)

    if width is not None and width > threashold:
        nIntersectedLines.append('right')
        if new_x_horizontal is not None:
            new_x_horizontal = (new_x_horizontal + point[0]) / 2
            new_y_horizontal = (new_y_horizontal + point[1]) / 2


    temp, width, pos, point = get_perpendicular_line(image, mask, x, y, (angle + np.pi), forward_distance)

    if width is not None and width > threashold:
        nIntersectedLines.append('down')
        if new_x_vertical is not None:
            new_x_vertical = (new_x_vertical + point[0]) / 2
            new_y_vertical = (new_y_vertical + point[1]) / 2
        
    if len(nIntersectedLines) > 3:
        return image, nIntersectedLines, (int((new_x_horizontal + new_x_vertical) / 2), int((new_y_horizontal + new_y_vertical) / 2))
    elif 'right' in nIntersectedLines and 'left' in nIntersectedLines:
        return image, nIntersectedLines, (int(new_x_horizontal), int(new_y_horizontal))
    elif 'down' in nIntersectedLines and 'up' in nIntersectedLines:
        return image, nIntersectedLines, (int(new_x_vertical), int(new_y_vertical))
    else:
        return image, nIntersectedLines, (None, None)

def get_perpendicular_line(image, mask, x, y, angle, forward_distance):
    points = []
    values = []
    num_points = 20
    length = 400
    step_size = length / num_points
    mask_height, mask_width = mask.shape[:2]

    for i in range(num_points):
        perp_dist = step_size * i - (step_size * num_points * 0.5)
        new_angle = np.arctan(perp_dist / forward_distance) + angle
        diagonal_distance = np.sqrt(pow(perp_dist, 2) + pow(forward_distance, 2))
        points.append((int(x + (np.sin(new_angle) * diagonal_distance)), int(y - (np.cos(new_angle) * diagonal_distance))))
        
        if (0 <= points[i][1] < mask_height) and (0 <= points[i][0] < mask_width):
            values.append(mask[points[i][1], points[i][0]])
        else:
            #return image, None, None, None  #Outside of frame
            values.append(0)
            

    top_black = None
    bottom_black = None
    for i, value in enumerate(values):
        if value is not None and value > 0:
            if top_black is None:
                top_black = i + 1  # First black pixel found (top of the line)
            else:
                bottom_black = i + 1  # Update with the latest black pixel (bottom of the line)
        cv2.circle(image, points[i], 5, (0, 0, 255), -1)

    if top_black is not None and bottom_black is not None:

        width = bottom_black - top_black + 1
        pos = (top_black + bottom_black) // 2  # Midpoint of the line
        
        # Draw circles at the edges of the black line
        '''
        if top_black_point:
            cv2.circle(image, top_black_point, 5, (255, 0, 0), -1)  # Green circle for top edge
        if bottom_black_point:
            cv2.circle(image, bottom_black_point, 5, (255, 0, 0), -1)  # Red circle for bottom edge
        '''
        return image, width, pos, points[pos - 1]
    else:
        return image, None, None, None  # No black line found

def locate_green_spaces(gMask):

    # Find contours in the mask
    contours, _ = cv2.findContours(gMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours if needed
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append((cX, cY))
    return points

while True:
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if errors > 10:
        break
    ret = False
    if devMode:
        ret, frame = cap.read()
    else:
        frame = picam2.capture_array()
        ret = True
    if ret:
        if not devMode:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame_bgr)
        #time.sleep(1)
        errors = 0
        space = False
        turnAround = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the mask (tweak these values as needed)

        ##Black Maska
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 50])
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        
        ##Green mask
        lower_green = np.array([50, 100, 40])  # Adjust these values as needed
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(frame, lower_green, upper_green)

        combined_mask = cv2.bitwise_or(green_mask, mask)

        # Apply the mask to the image
        frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
        
        forward_distance = 70  # Move 30 pixels forward before reading the line
        frame_height, frame_width = frame.shape[:2]  # Get the height and width of the frame
        x = None
        y = int(frame_height - 1 + (forward_distance / 2))
        angle = 0  # Line perpendicular to the bottom
        processed_frame = frame
        if lastX - 200 > 0 and lastX + 200 < frame_width - 1:
            for k in range(0, frame_height - 1, 20):
                for i in range(lastX - 200, lastX + 200, 2):
                    if mask[frame_height - k - 1, i] > 0:
                        if x is None:
                            x = i
                    elif x is not None:
                        x = int((i + x) / 2)
                        lastX = x# ----If i want it to adapt to previous line instead of center
                        break
                if x is not None:
                    break
        
        if x is None:
            for k in range(0, frame_height - 1, 20):
                for i in range(0, frame_width, 20):
                    if mask[frame_height - k - 1, i] > 0:
                        if x is None:
                            x = i
                    elif x is not None:
                        x = int((i + x) / 2)
                        lastX = x #----If i want it to adapt to previous line instead of center
                        break
                if x is not None:
                    break
        

        width = 0
        pos = 0
        graph_x = []
        graph_y = []
        if x is not None: 
            i = 0
            fitLine = True
            tried_intersection = False
            while pos is not None and i < 8:
                try:
                    processed_frame, width, pos, point = get_perpendicular_line(frame, mask, x, y, angle, forward_distance)
                    if pos is None:
                        processed_frame, width, pos, point = get_perpendicular_line(frame, mask, x, y, angle, forward_distance * 3)
                except:
                    continue
                nIntersectedLines = []

                if width is not None and width > 12 and tried_intersection is False:
                    tried_intersection = True
                    try:
                        linearReg = np.polyfit(graph_x, graph_y, 1)
                        #print(linearReg[0])
                        processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, point[0], point[1], np.arctan(linearReg[0]), 200)

                    except:
                        processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, point[0], point[1], angle, 200)
                        

                    if len(nIntersectedLines) > 2 and 'down' in nIntersectedLines:
                        cv2.circle(processed_frame, (intersection_point[0], intersection_point[1]), 30, (0, 0, 255), -1)
                        x = intersection_point[0]
                        y = intersection_point[1]
                        green_points = locate_green_spaces(green_mask)
                        temp = []
                        for p in green_points:
                            cv2.circle(processed_frame, (p[0], p[1]), 30, (255, 0, 0), -1)
                            if p[1] > y and p[0] > x and 'right' not in temp:
                                temp.append('right')
                            if p[1] > y and p[0] < x and 'left' not in temp:
                                temp.append('left')
                            if len(temp) > 1:
                                break
                            
                        if 'left' in temp and 'right' in temp:
                            angle = angle + (np.pi)
                            turnAround = True
                            cv2.line(processed_frame, (x, y), (int(graph_y[0]), int(-graph_x[0] + (forward_distance / 2) + (frame_height - 1))), color=(0, 0, 255), thickness=5)
                            cv2.circle(processed_frame, (x, y), 30, (0, 0, 255), -1)
                            break
                            #Make the line turn around
                        elif 'left' in temp:
                            angle = angle - (np.pi / 2)
                        elif 'right' in temp:
                            angle = angle + (np.pi / 2)



                        processed_frame, width, pos, point = get_perpendicular_line(frame, mask, x, y, angle, forward_distance * 3)
                        if pos is not None and point[0] is not None:
                            graph_x.append(((frame_height - 1) - y + (forward_distance / 2)))
                            graph_y.append((x - frame_width - 1))
                            if point[1] - y < 0:
                                angle = -np.arctan((point[0] - x) / (point[1] - y))
                            elif point[1] - y > 0:
                                angle = np.pi - np.arctan((point[0] - x) / (point[1] - y))
                            

                            x = x + int((point[0] - x) / 2)
                            y = y + int((point[1] - y) / 2)
                    elif len(nIntersectedLines) > 1 and 'down' not in nIntersectedLines:
                        fitLine = False


                #print(width)
                i = i + 1
                if pos is not None and len(nIntersectedLines) < 3:
                    graph_x.append(((frame_height - 1) - y + (forward_distance / 2)))
                    graph_y.append((x - frame_width - 1))
                    if point[1] - y < 0:
                        angle = -np.arctan((point[0] - x) / (point[1] - y))
                    elif point[1] - y > 0:
                        angle = np.pi - np.arctan((point[0] - x) / (point[1] - y))

                    x = x + int((point[0] - x) / 2)
                    y = y + int((point[1] - y) / 2)
            
            diagonal_distance = np.sqrt(pow(0, 2) + pow(forward_distance, 2))
            
            if pos is None and y > 400 and turnAround is False and 0 < int(x + (np.sin(angle) * diagonal_distance)) < frame_width - 1 and 0 < int(y - (np.cos(angle) * diagonal_distance)) < frame_height - 1:
                try:
                    linearReg = np.polyfit(graph_x, graph_y, 1)
                    #print(linearReg[0])
                except:
                    continue
                processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, x, y, np.arctan(linearReg[0]), 200)
                #cv2.circle(processed_frame, (x, y), 30, (255, 0, 0), -1)

                if ('down' in nIntersectedLines and len(nIntersectedLines) == 1) or len(nIntersectedLines) == 0:
                    if linearReg is not None:
                        polynomial = np.poly1d(linearReg)

                        x_fit = np.linspace(0, frame_height - 1, 100)
                        y_fit = polynomial(x_fit)
                        x_fit = np.linspace(frame_height - 1, 0, 100)
                        
                        points = np.array([(int(x), int(y)) for x, y in zip(y_fit, x_fit)], dtype=np.int32)
                        cv2.polylines(processed_frame, [points], isClosed=False, color=(255, 0, 0), thickness=5)
                        space = True


        #time.sleep(0.3)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            plt.scatter(graph_x, graph_y)
            plt.show()
        #time.sleep(100)
        if len(graph_x) > 0 and space is False and turnAround is False:
            try:
                if fitLine:
                    prevfit = fit
                    fit = np.polyfit(graph_x, graph_y, 2)
                    if prevfit is not None:
                        fit[0] = (fit[0] + prevfit[0]) / 2
                        fit[1] = (fit[1] + prevfit[1]) / 2
                if fit is not None:
                    polynomial = np.poly1d(fit)

                    '''If I only want to display "discouvered area"
                    y_fit = np.linspace(0, frame_height - 1 - (frame_height - 1 - int(max(graph_x))), 100)
                    x_fit = polynomial(y_fit)
                    y_fit = np.linspace(frame_height - 1, frame_height - 1 - int(max(graph_x)), 100)
                    #print(max(graph_y))
                    '''

                    #'''If I want the whole line:
                    y_fit = np.linspace(0, frame_height - 1, 100)
                    x_fit = polynomial(y_fit)
                    y_fit = np.linspace(frame_height - 1, 0, 100)
                    #'''
                    
                    points = np.array([(int(x), int(y)) for x, y in zip(x_fit, y_fit)], dtype=np.int32)
                    cv2.polylines(processed_frame, [points], isClosed=False, color=(0, 0, 255), thickness=5)

            except:
                continue
            
        for i in range(frame_height):
            continue
        frame_count += 1
        if frame_count >= 5:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Get the latest point in graphx and graphy
        if x and y:  # Ensure lists are not empty
            x_point = int(x)  # Latest point (highest index)
            y_point = int(y)
            #print(x_point, y_point)
            cv2.circle(processed_frame, (x_point, y_point), 30, (0, 0, 255), -1)

            # Calculate the coordinates of the lower middle point
            lower_middle_x = frame_width / 2

            # Calculate the differences
            dx = x_point - lower_middle_x
            
            dy = frame_height - y_point

            # Calculate the angle in degrees
            DrivingAngle = math.degrees(math.atan2(dy, dx))
            if DrivingAngle > 90:
                DrivingAngle = -180 + DrivingAngle
            
            #DrivingAngle = DrivingAngle * 0.1
            '''
            if DrivingAngle > 0:
                DrivingAngle = 0.0001 *pow(int(DrivingAngle), 3)
            else:
                DrivingAngle = -0.0001 * pow(abs(int(DrivingAngle)), 3)
            '''
            if DrivingAngle > 0:
                DrivingAngle = -(200/(DrivingAngle-91))
            else:
                DrivingAngle = (200/(DrivingAngle-91))
                
            print(DrivingAngle)
                
            if not devMode:
                runMotors(DrivingAngle, 40)
                
            cv2.putText(processed_frame, f"ANGLE: {DrivingAngle:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Lines Detection', processed_frame)

        if 0xFF == ord('q'):
            time.sleep(100)
            break
        #time.sleep(0.02)
    else:
        errors = errors + 1
