import numpy as np
import cv2

#loading haarcascade classifier
face_cascade = cv2.CascadeClassifier('src/cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	#Capture frame by frame
	ret, frame = cap.read()
	
	#gray color frames required for opencv to detect faces
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#detecting face in the gray color frame
	faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		print(x,y,w,h)
		
	if ret==True:
		#display the captured frame
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
	
	