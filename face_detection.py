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
		
		#region of interest
		roi = gray_frame[y:y+h,x:x+w] #[ycord_start:ycord_end]
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
		cv2.rect(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
		
		
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
	
	