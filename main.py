import os
import time

import cv2
import numpy as np
import torch

from custom_facenet_pytorch import Facenet
from retina_face import Retina_Face

# from facenet import Facenet


# cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture("http://192.168.100.170:4747/video")
database_directory = "./database"

prev_frame_time = 0
new_frame_time = 0

retinaface = Retina_Face()
facenet = Facenet()

while True:
    ret, frame = cap.read()

    # frame = cv2.imread("./sam.jpeg")
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

    if faces:
        for face in faces:
            embedding = facenet.get_face_embeddings(face)
            print(embedding.shape)
            for person in os.listdir(database_directory):
                for saved_embedding in os.listdir(
                    os.path.join(database_directory, person)
                ):
                    embedding_path = os.path.join(
                        database_directory, person, saved_embedding
                    )
                    embedding_loaded = np.load(embedding_path)
                    print(embedding_loaded.shape)
                    embedding_loaded = torch.from_numpy(embedding_loaded).to(
                        facenet.get_device()
                    )

                    # embedding_loaded = F.to_tensor(np.float32(face)).to(
                    #     facenet.get_device()
                    # )
                    print(embedding_loaded.shape)
                    distance = facenet.compare_faces(embedding, embedding_loaded)
                    print(distance)
                    if distance < 1:
                        print(person)
                        break

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

# frame = cv2.imread("./sib.jpeg")
# annotations = retinaface.get_annotations(frame)
# faces = retinaface.extract_faces(frame, annotations)
# if faces:
#     for face in faces:
#         embedding = facenet.get_face_embeddings(face)
#         for person in os.listdir(database_directory):
#             for saved_embedding in os.listdir(os.path.join(database_directory, person)):
#                 embedding_path = os.path.join(
#                     database_directory, person, saved_embedding
#                 )
#                 embedding_loaded = np.load(embedding_path)
#                 distance = facenet.compare_faces(embedding, embedding_loaded)
#                 if distance < 0.5:
#                     print(person)
#                     break
