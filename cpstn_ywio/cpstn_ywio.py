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

#pink (daytime)
# L: 155, 130, 165
# U: 184, 237, 245

# Color ranges for the ball (adjust as needed)
lower_color_range = np.array([155, 130, 165])
upper_color_range = np.array([184, 237, 245])

cap = cv2.VideoCapture(0)

prev_cx = None
fps = cap.get(cv2.CAP_PROP_FPS)
speed = 0
cx = 0

predicted_x = None

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color_range, upper_color_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        ball_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(ball_contour)
        cx = int(M['m10'] / M['m00'])

        if prev_cx is not None:
            distance_traveled = cx - prev_cx  # Signed horizontal distance
            speed = (distance_traveled * fps) / 5  # Approximate speed (adjust the divisor if needed)
            cv2.putText(frame, f"X-Coord: {cx}, Speed: {speed:.2f} px/s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        prev_cx = cx

    # Read time data
    with open("time_data.txt", "r") as f:
        try:
            time_remaining = float(f.read().strip())
        except ValueError:
            time_remaining = 0  # Default to 0 if the file content is invalid

    # Calculate predicted X only if time_remaining is not 0
    if time_remaining > 0:
        predicted_x = cx + (speed * time_remaining)

        # Write predicted_x to output file
        with open("predicted_x.txt", "w") as f:
            f.write(f"{predicted_x:.2f}")

        # Reset time_data.txt to 0
        with open("time_data.txt", "w") as f:
            f.write("0")

    # Display predicted X only when available
    if predicted_x is not None:
        cv2.putText(frame, f"Predicted X: {predicted_x:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.circle(frame, (cx, 100), 5, (0, 0, 255), -1)  # Visual marker at the center

    cv2.imshow('Frame', frame)
    if cv2.waitKey(250) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
