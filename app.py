from flask import Flask, Response, redirect, render_template, request, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime
import time
from PIL import Image
import numpy as np
import cv2
import os
import json


# Openning config.json:*********************
with open('templates/config.json', 'r') as c:
    params = json.load(c)["params"]


# Flask Setup ******************************
app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'faceRecognition'


# Sqlalchemy Setup *********************
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app.config["SQLALCHEMY_DATABASE_URI"] = params["local_uri"]
db.init_app(app)


class StudentDetail(db.Model):
    username: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=False, nullable=True)
    password: Mapped[str] = mapped_column(unique=True, nullable=True)
    date: Mapped[datetime] = mapped_column(unique=False, nullable=True)
    img_id: Mapped[int] = mapped_column(unique=True, nullable=True)


# Cv2 setup:
face_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), "Haarcascades", "haarcascade_frontalface_default.xml"))
cap = cv2.VideoCapture(0)

# importent function******************

def generate_dataset(img_id):
    try:
        os.makedirs("faces/" + img_id)
    except:
        pass
    global face_cascade
    global cap
    img_count = 1
    while True:
        ret, img = cap.read()
        if ret:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            if faces is not ():
                face = None
                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    
                file_name_path = os.path.join(os.getcwd(), "faces", img_id, f"{img_count}.jpg")
                cv2.imwrite(file_name_path, face)
                img_count += 1

                frame = cv2.imencode('.jpg', face)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                

        if cv2.waitKey(1) == 13 or img_count >= 401:
            cap.release()
            cv2.destroyAllWindows()
            break

def TakeAttendance(min):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("classifier.xml")
    global face_cascade
    global cap
    t_end = time.time() + 60 * min

    while time.time() < t_end:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                
                if conf >= 45:
                    print(id_)
                    print("__")



                color = (255, 0, 0)
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

    cap.release()
    cv2.destroyAllWindows()


# Routes: ******************************
@app.route('/')
def home():
    return render_template("index.html", params=params)


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    if 'admin' in session and session['admin'] == params['admin_user']:
        return render_template("dashboard.html", params=params, admin_session=True)

    if request.method == 'POST':
        username = request.form.get('username')
        userpass = request.form.get('password')
        if (username == params['admin_user'] and userpass == params['admin_password']):
            session['admin'] = username
            return render_template("dashboard.html", params=params, admin_session=True)
        else:
            pass
            # return render_template('login.html', params=params, wrongEntry=True)

    return render_template('login.html', params=params)

@app.route('/dashboard/add-student', methods=['GET', 'POST'])
def addStudent():
    if 'admin' in session and session['admin'] == params['admin_user']:
        if (request.method == 'POST'):
            username = request.form.get('username').lower()
            fullname = request.form.get('fullname')
            password = request.form.get('password')
            confirmpassword = request.form.get('confirmpassword')

            if password == confirmpassword:
                entry = StudentDetail(
                    username=username, name=fullname, password=password, date=datetime.now())
                db.session.add(entry)
                db.session.commit()
                img_id = StudentDetail.query.filter_by(username=username).first().img_id
                
                return render_template("capture_img.html", params=params, username=img_id)
            else:
                pass

        return render_template("add_student.html", params=params)
    return redirect("/dashboard")

@app.route('/vidfeed_captureimg/<string:username>')
def vidfeed_captureimg(username):
    if 'admin' in session and session['admin'] == params['admin_user']:
        return Response(generate_dataset(username), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect("/dashboard")

@app.route('/train_classifier')
def train_classifier():
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    dataset_dir = os.path.join(os.getcwd(), "faces")
    faces = []
    ids = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            # print(os.path.basename(root)) 
            image = os.path.join(root, file)
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id_ = os.path.basename(root)

            faces.append(imageNp)
            ids.append(int(id_))
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")

    return redirect('/dashboard')

@app.route('/take_attendance/<int:time>')
def take_attendance(time):
    if 'admin' in session and session['admin'] == params['admin_user']:
        return render_template("take_attendance.html", params=params, time=time)

@app.route('/start_attendance/<int:time>')
def start_attendance(time):
    if 'admin' in session and session['admin'] == params['admin_user']:
        return Response(TakeAttendance(time), mimetype='multipart/x-mixed-replace; boundary=frame')
    return redirect("/dashboard")

@app.route('/dashboard/logout')
def logout():
    user = request.args.get('user')
    if user == 'admin':
        if 'admin' in session and session['admin'] == params['admin_user']:
            session.pop("admin")
            return redirect('/dashboard')
    elif user == "student":
        pass
    return redirect('/dashboard')


if __name__ == "__main__":
    app.run(debug=True)
