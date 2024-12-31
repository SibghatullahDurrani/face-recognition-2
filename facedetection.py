import time

import cv2
import numpy as np

from retina_face import Retina_Face

cap = cv2.VideoCapture("http://192.168.100.170:4747/video")
database_directory = "./database"

prev_frame_time = 0
new_frame_time = 0

retinaface = Retina_Face()

while True:
    ret, frame = cap.read()

    # frame = cv2.imread("./sam.jpeg")
    # annotations = retinaface.get_annotations(frame)
    # for face_annotation in annotations:
    #     bbox: list[float]
    #     bbox = face_annotation.get("bbox")
    #     score: np.float32
    #     score = face_annotation.get("score")
    #     if score > 0.8:
    #         frame = cv2.rectangle(
    #             img=frame,
    #             pt1=(int(bbox[0]), int(bbox[1])),
    #             pt2=(int(bbox[2]), int(bbox[3])),
    #             color=(255, 0, 0),
    #             thickness=1,
    #         )
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
