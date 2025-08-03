import cv2
import numpy as np

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(f'Module_04_Face_Detection/assests/haarcascade_frontalface_default.xml')

# Load an image from a file (e.g. common face library)
img = cv2.imread(f'Module_04_Face_Detection/assests/Viola_Davis_0001.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image using the Haar Cascade Classifier
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

# Draw rectangles around each detected face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the original image with rectangles drawn around each detected face
cv2.imshow('Original Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load a second image from a file (e.g. another common face library image)
img2 = cv2.imread('Module_04_Face_Detection/assests/Jim_Carrey_0001.jpg')

# Convert the second image to grayscale
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect faces in the second image using the Haar Cascade Classifier
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

# Draw rectangles around each detected face in the second image
for (x, y, w, h) in faces2:
    cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the second image with rectangles drawn around each detected face
cv2.imshow('Second Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()