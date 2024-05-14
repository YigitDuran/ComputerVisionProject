import cv2


def display_hsv_value(event, x, y, flags, param):
    global hsv_image

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV Value at ({x}, {y}): {hsv_image[y, x]}")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow('Frame', frame)
    cv2.setMouseCallback('Frame', display_hsv_value)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
