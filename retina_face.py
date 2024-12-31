import numpy as np
from retinaface.pre_trained_models import get_model


class Retina_Face:
    model = get_model("resnet50_2020-07-20", 2048, device="cuda")
    model.eval()

    def get_annotations(self, img):
        return self.model.predict_jsons(img)

    def extract_faces(self, img, annotations):
        imgs: list[np.ndarray] = []
        for face_annotation in annotations:
            bbox: list[float]
            bbox = face_annotation.get("bbox")
            score: np.float32
            score = face_annotation.get("score")
            if score > 0.8:
                imgs.append(img[bbox[1] : bbox[3], bbox[0] : bbox[2]])

            return imgs
