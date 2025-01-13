import cv2
import numpy as np
from math import cos, sin, radians
import time

def calculate_new_point(x, y, angle_degrees, distance):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Calculate new coordinates
    x2 = x + distance * np.cos(angle_radians)
    y2 = y + distance * np.sin(angle_radians)

    return int(round(x2)), int(round(y2))


def sample_line(image, x, y, angle, length, num_points):
    """
    Samples pixel values along a line extending from a point at a given angle and distance.
    
    Parameters:
    - image: The image array (HSV format).
    - x, y: Starting point (coordinates).
    - angle: The angle in degrees relative to the horizontal.
    - length: The total length of the line to sample.
    - num_points: The number of points to sample along the line.
    
    Returns:
    - pixel_values: The pixel values along the sampled line.
    - sampled_points: The coordinates of the sampled points.
    """
    pixel_values = []
    sampled_points = []

    # Step size based on the number of points and length of the line
    step_size = length / num_points

    # Convert the angle to radians
    rad = radians(angle)

    # Loop through the number of points to sample along the line
    for i in range(num_points):
        # Calculate the new x and y positions based on the distance and angle
        distance = i * step_size - (step_size * 0.5 * num_points)
        new_x = int(x + distance * cos(rad))
        new_y = int(y + distance * sin(rad))
        
        # Ensure the new coordinates are within the image boundaries
        if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0]:
            # Read the pixel value from the image
            pixel_value = image[new_y, new_x]  # Note: image[y, x] since row comes first
            pixel_values.append(pixel_value)
            sampled_points.append((new_x, new_y))
        else:
            pixel_values.append(None)
            sampled_points.append((None, None))  # Point is out of bounds

    return pixel_values, sampled_points

def is_black(hsv_value, v_threshold=70, s_threshold=100):
    """Check if the HSV value corresponds to black."""
    if hsv_value is None:
        return False
    h, s, v = hsv_value
    # Check if the pixel is dark (low value) and not saturated
    return v != 0 and s != 0  # Adjust thresholds for better accuracy

def get_line_width_and_position(image, x, y, angle, length=100, num_points=100, threshold=30):
    """
    Calculates the width and position of a black line along a continuous sampled line
    and draws circles where the black line edges are found.
    
    Parameters:
    - image: The image array (HSV format).
    - x, y: Starting point (coordinates).
    - angle: The angle in degrees relative to the horizontal.
    - length: The total length of the line to sample (default 100).
    - num_points: Number of points to sample along the line (default 100).
    - threshold: The HSV value threshold to define black (default 30).
    
    Returns:
    - line_width: Width of the black line.
    - line_position: Position (midpoint) of the black line.
    """
    # Sample the line for pixel values
    pixel_values, sampled_points = sample_line(image, x, y, angle, length, num_points)
    
    # Initialize variables to store top and bottom black edges
    top_black = None
    bottom_black = None
    top_black_point = None
    bottom_black_point = None
    # Traverse the sampled pixel values to find black points
    for i, (pixel_value, point) in enumerate(zip(pixel_values, sampled_points)):
        if pixel_value is not None and is_black(pixel_value, threshold):
            if top_black is None:
                top_black = i  # First black pixel found (top of the line)
                top_black_point = point  # Save coordinates for circle
            else:
                bottom_black = i  # Update with the latest black pixel (bottom of the line)
                bottom_black_point = point  # Save coordinates for circle

    # Calculate the width and position of the black line
    if top_black is not None and bottom_black is not None:

        line_width = bottom_black - top_black + 1
        line_position = (top_black + bottom_black) // 2  # Midpoint of the line
        
        # Draw circles at the edges of the black line
        if top_black_point:
            cv2.circle(image, top_black_point, 5, (0, 255, 0), -1)  # Green circle for top edge
        if bottom_black_point:
            cv2.circle(image, bottom_black_point, 5, (0, 0, 255), -1)  # Red circle for bottom edge
        
        return line_width, line_position
    else:
        return None, None  # No black line found

def get_perpendicular_line(image, x, y, angle, forward_distance, length=100, num_points=100):
    """
    Moves forward in a given direction by a certain distance, then samples a perpendicular line.
    
    Parameters:
    - image: The image array (HSV format).
    - x, y: Starting point (coordinates).
    - angle: The direction to move forward in degrees.
    - forward_distance: The distance to move forward before sampling the perpendicular line.
    - length: The length of the perpendicular line to sample.
    - num_points: Number of points to sample along the perpendicular line.
    
    Returns:
    - line_width: Width of the black line found in the perpendicular line.
    - line_position: Midpoint of the black line.
    """
    # Move forward by the specified distance in the given angle direction
    rad = radians(angle)
    x_new = int(x + forward_distance * cos(rad))
    y_new = int(y + forward_distance * sin(rad))

    # Sample a perpendicular line to the original angle (+90 degrees)
    perpendicular_angle = angle + 90  # Adjust for the perpendicular direction
    pixel_values, sampled_points = sample_line(image, x_new, y_new, perpendicular_angle, length, num_points)
    
    # Calculate the start and end points for the line to be drawn
    start_x = int(x_new + (length / 2) * cos(radians(perpendicular_angle)))
    start_y = int(y_new + (length / 2) * sin(radians(perpendicular_angle)))
    end_x = int(x_new - (length / 2) * cos(radians(perpendicular_angle)))
    end_y = int(y_new - (length / 2) * sin(radians(perpendicular_angle)))
    
    # Draw the sampled perpendicular line
    #cv2.line(processed_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)  # Draw blue line for the sampled line

    # Proceed to calculate black line width/position
    return get_line_width_and_position(image, x_new, y_new, perpendicular_angle, length, num_points)
# Open the video
cap = cv2.VideoCapture('TestRun.mp4')

# Create a named window
cv2.namedWindow('Lines Detection', cv2.WINDOW_NORMAL)

# Move the window to the desired position (x, y)
window_x = 100  # Horizontal position
window_y = 100  # Vertical position
cv2.moveWindow('Lines Detection', window_x, window_y)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds for the mask (tweak these values as needed)
    lower_bound = np.array([0, 0, 0])  # Adjust based on desired color range
    upper_bound = np.array([180, 255, 70])  # Example for detecting dark areas

    # Create a binary mask
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    
    # Apply the mask to the image
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    forward_distance = 300  # Move 30 pixels forward before reading the line
    frame_height, frame_width = frame.shape[:2]  # Get the height and width of the frame
    x = None
    y = int(frame_height - 1 + (forward_distance / 3))
    angle =  -90.0  # Line perpendicular to the bottom
    length = 300  # Length of the perpendicular line to sample
    num_points = 100  # Number of points to sample along the perpendicular line
    processed_frame = frame

    for i in range(0, frame_width, 20):
        if mask[frame_height - 1, i] > 0:
            if x is None:
                x = i
            else:
                x = int((i + x) / 2)
                break

    
    # Get the width and position of the black line
    width = None
    pos = None
    if x is not None: 
        width, pos = get_perpendicular_line(frame, x, y, angle, forward_distance, length, num_points)
        while pos is not None:
            width, pos = get_perpendicular_line(frame, x, y, angle, forward_distance, length, num_points)
            
            if pos is not None:
                #angle = angle - 0.01*(300-pos)
                x2, y2 = calculate_new_point(x, y, angle, forward_distance)
                distance = i * step_size - (step_size * 0.5 * num_points)
                new_x = int(x + distance * cos(rad))
                new_y = int(y + distance * sin(rad))

            
            x, y = calculate_new_point(x, y, angle, 50)
            print(f"Line width: {width}, Line position: {pos}, Angle {angle}, x {x}, y {y}")
            
            #cv2.imshow('Lines Detection', processed_frame)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #time.sleep(0.05)

    #time.sleep(100)
    
    
    cv2.imshow('Lines Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.3)

cap.release()
cv2.destroyAllWindows()