import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")
print(image_dir)

#loading haarcascade classifier
face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []  #actual number related to the labels or rather names
x_train = []   #holds actual pixel values

#Traversing directories to look for images and grab the label names.
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
 			path = os.path.join(root, file)
 			label = os.path.basename(os.path.dirname(path))

 			#adding labels for each person's faces
 			if not label in label_ids:
 				label_ids[label] = current_id
 				current_id += 1

 			id_ = label_ids[label]	
 			print(label_ids)

 			pil_image = Image.open(path).convert("L") #grayscale image
 			image_array = np.array(pil_image, "uint8") #putting image to numpy array

 			#detecting face in the gray color frame
 			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

 			for(x,y,w,h) in faces:
 				roi = image_array[y:y+h, x:x+w]
 				x_train.append(roi)
 				y_labels.append(id_)

 #save labels in order to use them in face_detection.py
with open("labels.pickle",'wb') as f:
 	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels)) #training data, labels into numpy array
recognizer.save("trainer.yml")	