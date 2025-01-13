import cv2
import numpy as np
from math import cos, sin, radians
import math
import time
import matplotlib.pyplot as plt

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
    length = 300
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


# Open the video
cap = cv2.VideoCapture('IntersectionNewCamera.mp4')
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
while True:
    if errors > 10:
        break
    ret, frame = cap.read()

    if ret:
        errors = 0
        space = False
        turnAround = False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds for the mask (tweak these values as needed)

        ##Black Mask
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 50])
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        
        ##Green mask
        #lower_green = np.array([50, 100, 40])  # Adjust these values as needed
        #upper_green = np.array([70, 255, 255])

        lower_green = np.array([50, 100, 40])  # Adjust these values as needed
        upper_green = np.array([70, 255, 255])
        green_mask = cv2.inRange(frame, lower_green, upper_green)

        #combined_mask = cv2.bitwise_or(green_mask, mask)

        # Apply the mask to the image
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        forward_distance = 100  # Move 30 pixels forward before reading the line
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
                        #lastX = x ----If i want it to adapt to previous line instead of center
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
                        #lastX = x ----If i want it to adapt to previous line instead of center
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
            while pos is not None and i < 20:
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
                        print(linearReg[0])
                        processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, point[0], point[1], np.arctan(linearReg[0]), 300)

                    except:
                        processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, point[0], point[1], angle, 300)
                        

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
                            if point[1] - y is not 0:
                                angle = -np.arctan((point[0] - x) / (point[1] - y))
                            x = x + int((point[0] - x) / 2)
                            y = y + int((point[1] - y) / 2)
                    elif len(nIntersectedLines) > 1 and 'down' not in nIntersectedLines:
                        fitLine = False


                #print(width)
                i = i + 1
                if pos is not None and len(nIntersectedLines) < 3:
                    graph_x.append(((frame_height - 1) - y + (forward_distance / 2)))
                    graph_y.append((x))
                    if point[1] - y is not 0:
                        angle = -np.arctan((point[0] - x) / (point[1] - y))
                    x = x + int((point[0] - x) / 2)
                    y = y + int((point[1] - y) / 2)
            
            diagonal_distance = np.sqrt(pow(0, 2) + pow(forward_distance, 2))
            
            if pos is None and y > 400 and turnAround is False and 0 < int(x + (np.sin(angle) * diagonal_distance)) < frame_width - 1 and 0 < int(y - (np.cos(angle) * diagonal_distance)) < frame_height - 1:
                try:
                    linearReg = np.polyfit(graph_x, graph_y, 1)
                    print(linearReg[0])
                except:
                    continue
                processed_frame, nIntersectedLines, intersection_point = get_intersection_data(processed_frame, mask, x, y, np.arctan(linearReg[0]), 150)
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

        
        cv2.imshow('Lines Detection', processed_frame)

        if 0xFF == ord('q'):
            time.sleep(100)
            break
        time.sleep(0.02)
    else:
        errors = errors + 1
        

cap.release()
cv2.destroyAllWindows()