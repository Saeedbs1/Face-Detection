import cv2
import numpy as np
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)
print("Face Detection running")
while True:
    
    # Capture frames
    rent, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

  
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
    
       cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 255, 255, 2))
        
   
    # Display the resulting frame

    cv2.imshow("Face Detection", frames)

    #Press q to stop the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
