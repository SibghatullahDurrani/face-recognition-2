import os
import uuid

import cv2
import numpy as np

from custom_facenet_pytorch import Facenet
from custom_mtcnn import Custom_MTCNN

# from facenet import Facenet
from retina_face import Retina_Face

register_images_directory = "./register_images"
database_directory = "./database"

retinaface = Retina_Face()
mtcnn = Custom_MTCNN()
facenet = Facenet()


def register_images(face_detector):
    for images in os.listdir(register_images_directory):
        image_path = os.path.join(register_images_directory, images)
        img = cv2.imread(image_path)
        faces = []
        if face_detector == "retinaface":
            face_annotations = retinaface.get_annotations(img)
            faces = retinaface.extract_faces(img, face_annotations)
        elif face_detector == "mtcnn":
            face_annotations, _ = mtcnn.get_annotations(img)
            faces = mtcnn.extract_faces(img, face_annotations)
        count = 0
        generated_uuid = str(uuid.uuid4())
        where_to_save = (
            os.path.join(database_directory, generated_uuid) + "/" + str(count)
        )
        if faces:
            for face in faces:
                embedding = facenet.get_face_embeddings(face[0])
                embedding = embedding.cpu()
                embedding = embedding.numpy()
                try:
                    np.save(where_to_save, embedding)
                except:
                    os.mkdir(os.path.join(database_directory, generated_uuid))
                    np.save(where_to_save, embedding)

                count = count + 1


register_images("mtcnn")
