import cv2

pipeline = (
    "udpsrc port=5000 ! "
    "application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
print("isOpened:", cap.isOpened())
while True:
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.imshow("UDP Stream", frame)
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()