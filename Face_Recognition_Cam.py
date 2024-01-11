import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_coordinates = trained_face_data.detectMultiScale(grayscaled_img)



    #Draw rectangles around the faces
    for (x, y, w, h) in faces_coordinates :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Clever Programmer face Detector", frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()


print("Code Completed")