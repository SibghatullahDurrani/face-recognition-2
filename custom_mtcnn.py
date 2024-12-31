from facenet_pytorch import MTCNN


class Custom_MTCNN:
    def get_annotations(self, img):
        mtcnn = MTCNN()
        return mtcnn.detect(img)

    def extract_faces(self, img, annotations):
        imgs = []
        out = []
        if annotations is not None:
            for annotation in annotations:
                imgs.append(
                    img[
                        int(annotation.item(1)) : int(annotation.item(3)),
                        int(annotation.item(0)) : int(annotation.item(2)),
                    ]
                )
                imgs.append(int(annotation.item(0)))
                imgs.append(int(annotation.item(1)))
                out.append(imgs)
                imgs = []

            return out
