import numpy as np
import cv2
import os

name = input("What is first your name?")
print("Nice meeting you {}!".format(name))

print("Press 'K' when you are ready")

total = 0

path = os.path.join("images", name)
if not os.path.exists(path):
	os.makedirs(path)

print("Directory '%s' created" %path) 
 
#loading haarcascade classifier
face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	#Capture frame by frame
 	ret, frame = cap.read()
 	orig = frame.copy()
 	#gray color frames required for opencv to detect faces
 	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 	#detecting face in the gray color frame
 	faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
 	for (x, y, w, h) in faces:
 		print(x,y,w,h)

 		#region of interest
 		roi_gray = gray_frame[y:y+h,x:x+w] #[ycord_start:ycord_end]
 		roi_color = frame[y:y+h,x:x+w]
 		img_file = "image.png"

 		#printing out grayed roi for future reference
 		cv2.imwrite(img_file, roi_gray)

 		#drawing a rectangle around a detected face
 		color = (255,0,0)
 		stroke = 2
 		end_cord_x = x + w;
 		end_cord_y = y + h;
 		#starting & ending coordinates
 		cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)


 	if ret==True:
 		#display the captured frame
 		cv2.imshow('frame',frame)
 		key = cv2.waitKey(1) & 0xFF
 		if key == ord("k"):
 			#save_path = "images{}".format(name)
 			face_img = os.path.join(str(path),"{}.png".format(str(total).zfill(5))) 
 			cv2.imwrite(face_img, orig)
 			total += 1
 		elif key == ord('q'):
 			break
 	else:
 		break
# Release everything if job is finished
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")

cap.release()
cv2.destroyAllWindows()
