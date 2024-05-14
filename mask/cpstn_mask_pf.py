import cv2
import numpy as np

# old color range (red):
# L: 0, 100, 100
# U: 20, 255, 255

# New color range (pink):
# L: 162, 150, 175
# U: 174, 217, 225

# New color range (pink):
# L: 155, 150, 175
# U: 174, 217, 225

# Current color range (pink):
# L: 155, 130, 165
# U: 184, 237, 245

#pink (daytime)
# L: 155, 130, 165
# U: 184, 237, 245

# Color ranges for ball detection (will be fine-tuned)
lower_color_range = np.array([155, 130, 165])
upper_color_range = np.array([184, 240, 250])

# Setup video capture
cap = cv2.VideoCapture(0)

# Variables for speed calculation
prev_cx, prev_cy = None, None
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second

# Variables for trail effect (Might get removed as the final product, this is for debug purposes)
trail_length = 30  # Adjust for desired trail length
trail_points = []

while True:
    ret, frame = cap.read()

    # Ball detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color_range, upper_color_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        ball_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(ball_contour)
        cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])

        # Speed calculation (estimation)
        if prev_cx is not None and prev_cy is not None:
            distance_traveled = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            speed = (distance_traveled * fps) / trail_length  # Approximate speed
            cv2.putText(frame, f"Speed: {speed:.2f} px/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        prev_cx, prev_cy = cx, cy

        # Trail effect
        trail_points.append((cx, cy))
        if len(trail_points) > trail_length:
            trail_points.pop(0)

        for i in range(1, len(trail_points)):
            cv2.line(frame, trail_points[i-1], trail_points[i], (0, 0, 255), 2)

        # Display ball and coordinates
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Coordinates: ({cx}, {cy})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display both frame and mask
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
