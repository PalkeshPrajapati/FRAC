import os
from PIL import Image
import numpy as np
import cv2

dataset_dir = os.path.join(os.getcwd(), "faces")

faces = []
ids = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        # print(os.path.basename(root)) 
        image = os.path.join(root, file)
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id_ = os.path.basename(root)

        faces.append(imageNp)
        ids.append(int(id_))



# Train the classifier and save
clf = cv2.face.LBPHFaceRecognizer_create()
clf.train(faces, np.array(ids))
clf.write("classifier.xml")
