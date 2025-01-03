import os
import time

import cv2
import numpy as np
import torch

import database_operations
from custom_facenet_pytorch import Facenet
from custom_mtcnn import Custom_MTCNN
from retina_face import Retina_Face

# from facenet import Facenet


# cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture("http://192.168.100.165:4747/video")
database_directory = "./database"


retinaface = Retina_Face()
facenet = Facenet()
mtcnn = Custom_MTCNN()


def run(detector, should_recognize):
    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, frame = cap.read()

        faces = []
        annotations = []
        if detector == "retinaface":
            annotations = retinaface.get_annotations(frame)
            faces = retinaface.extract_faces(frame, annotations)
            for face_annotation in annotations:
                bbox: list[float]
                bbox = face_annotation.get("bbox")
                score: np.float32
                score = face_annotation.get("score")
                if score > 0.8:
                    frame = cv2.rectangle(
                        img=frame,
                        pt1=(int(bbox[0]), int(bbox[1])),
                        pt2=(int(bbox[2]), int(bbox[3])),
                        color=(255, 0, 0),
                        thickness=1,
                    )
        elif detector == "mtcnn":
            annotations, _ = mtcnn.get_annotations(frame)
            faces = mtcnn.extract_faces(frame, annotations)
            if annotations is not None:
                for box in annotations:
                    frame = cv2.rectangle(
                        img=frame,
                        pt1=(int(box.item(0)), int(box.item(1))),
                        pt2=(int(box.item(2)), int(box.item(3))),
                        color=(255, 0, 0),
                        thickness=1,
                    )

        if should_recognize == True and detector != "none":
            if faces:
                for face in faces:
                    embedding = facenet.get_face_embeddings(face[0])
                    for person in os.listdir(database_directory):
                        for saved_embedding in os.listdir(
                            os.path.join(database_directory, person)
                        ):
                            embedding_path = os.path.join(
                                database_directory, person, saved_embedding
                            )
                            embedding_loaded = np.load(embedding_path)
                            embedding_loaded = torch.from_numpy(embedding_loaded).to(
                                facenet.get_device()
                            )
                            distance = facenet.compare_faces(
                                embedding, embedding_loaded
                            )
                            if distance < 1:
                                employee_name = database_operations.get_employee_name(
                                    person
                                )
                                database_operations.manage_attendance(person)
                                cv2.putText(
                                    frame,
                                    employee_name,
                                    (face[1], face[2]),
                                    font,
                                    1,
                                    (100, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                                break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("video", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break


run(detector="mtcnn", should_recognize=True)
