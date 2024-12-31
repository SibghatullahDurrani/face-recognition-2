import cv2
import keras
import numpy as np
from numpy import linalg as LA


class Facenet:
    model = keras.models.load_model("./facenet_keras.h5", compile=False)

    def preprocess_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def get_face_embeddings(self, frame):
        img = self.preprocess_image(frame)
        return self.model.predict(img)

    def compare_faces(self, embedding1, embedding2):
        distance = LA.norm(embedding1 - embedding2)
        return distance
