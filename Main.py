from deepface import DeepFace
import cv2

# Known image path
known_image_path = "known.jpg"

# Open video
video = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Detect faces using OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            # Compare with known image
            result = DeepFace.verify(face_img, known_image_path)

            if result["verified"]:
                name = "Known Person"
            else:
                name = "Unknown"

        except:
            name = "Error"

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put text
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("DeepFace Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
