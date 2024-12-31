import cv2
from facenet_pytorch import MTCNN

# cap = cv2.VideoCapture("http://192.168.100.170:4747/video")
mtcnn = MTCNN()

while True:
    # ret, frame = cap.read()
    frame = cv2.imread("./sib2.jpeg")
    boxes, _ = mtcnn.detect(frame)
    print(boxes)
    if boxes is not None:
        for box in boxes:
            frame = cv2.rectangle(
                img=frame,
                pt1=(int(box.item(0)), int(box.item(1))),
                pt2=(int(box.item(2)), int(box.item(3))),
                color=(255, 0, 0),
                thickness=1,
            )
    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
