import os

import cv2
import numpy as np

from custom_facenet_pytorch import Facenet

# from facenet import Facenet
from retina_face import Retina_Face

register_images_directory = "./register_images"
database_directory = "./database"

retinaface = Retina_Face()
facenet = Facenet()

for images in os.listdir(register_images_directory):
    imageName = images.split(".")[0]
    image_path = os.path.join(register_images_directory, images)
    img = cv2.imread(image_path)
    face_annotations = retinaface.get_annotations(img)
    faces = retinaface.extract_faces(img, face_annotations)
    count = 0
    where_to_save = os.path.join(database_directory, imageName) + "/" + str(count)
    if faces:
        for face in faces:
            embedding = facenet.get_face_embeddings(face)
            print(embedding.shape)
            embedding = embedding.cpu()
            embedding = embedding.numpy()
            print(embedding.shape)
            try:
                np.save(where_to_save, embedding)
            except:
                os.mkdir(os.path.join(database_directory, imageName))
                np.save(where_to_save, embedding)

            print(where_to_save)
            count = count + 1

    # for face_annotation in face_annotations:
    #     bbox: list[float]
    #     bbox = face_annotation.get("bbox")
    #     score: np.float32
    #     score = face_annotation.get("score")
    #     if score > 0.8:
    #         cropped_img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    #         where_to_save = (
    #             os.path.join(database_directory, imageName) + "/" + str(count)
    #         )
    #         embedding = facenet.get_face_embeddings(cropped_img)
    #         embedding = embedding.cpu()
    #         embedding = embedding.numpy()
    #         try:
    #             np.save(where_to_save, embedding)
    #         except:
    #             os.mkdir(os.path.join(database_directory, imageName))
    #             np.save(where_to_save, embedding)
    #
    #         print(where_to_save)
    #         count = count + 1
