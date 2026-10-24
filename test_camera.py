import cv2

import config

cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW) # CAP_DSHOW is the Windows camera driver
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open the camera. Close Teams/Zoom/Camera app, or try CAMERA_INDEX = 1 in config.py")
    raise SystemExit

print("Camera opened. Press Q in the video window to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame.")
        break
    h, w = frame.shape[:2]
    cv2.putText(frame, f"{w}x{h} - press Q to quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Camera test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
