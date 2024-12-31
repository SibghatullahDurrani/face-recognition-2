import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import functional as F


class Facenet:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def preprocess_img(self, frame):
        if frame is not None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
            face = F.to_tensor(np.float32(face)).to(self.device)
            face = (face - 127.5) / 128.0
            face = face.unsqueeze(0)
            return face

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (160, 160))
        # img = img.astype("float32") / 255.0
        # img = np.expand_dims(img, axis=0)
        # return img

    def get_face_embeddings(self, frame):
        img = self.preprocess_img(frame)
        return self.resnet(img).detach()

    def compare_faces(self, embedding1, embedding2):
        return (embedding1 - embedding2).norm().item()

    def get_device(self):
        return self.device
