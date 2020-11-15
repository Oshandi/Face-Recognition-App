import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret, frame = cap.read()
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
	
	