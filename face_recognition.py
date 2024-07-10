import cv2
import os
import time




def TakeAttendance(min):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), "Haarcascades", "haarcascade_frontalface_default.xml"))
    cap = cv2.VideoCapture(0)
    t_end = time.time() + 60 * min

    while time.time() < t_end:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            id_, conf = recognizer.predict(roi_gray)
            
            if conf >= 45:
                if (id_ == 30):
                    print("Harshita Prajapti")
                elif (id_ == 31):
                    print("Palkesh Prajapti")
                elif (id_ == 33):
                    print("Pankaj Deshmukh")
                elif (id_ == 37):
                    print("Priyanka Shrivastava")
                print("__")



            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow("frame", frame)
        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

TakeAttendance(5)