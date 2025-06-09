import cv2
import zmq
import pickle

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    data = socket.recv()
    frame = pickle.loads(data)
    # Xử lý frame tùy ý, ví dụ hiển thị
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()