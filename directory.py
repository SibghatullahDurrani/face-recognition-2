import os

directory = "./database"

for person in os.listdir(directory):
    images_path = os.path.join(directory, person)
    for image in os.listdir(images_path):
        print(os.path.join(images_path, image))
