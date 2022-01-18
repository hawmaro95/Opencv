import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import flask
import psycopg2

 
path = 'ImagesAttendance'
images = []
classNames = []
classId = []
CheckIn = []
myList = os.listdir(path)
print(myList)
t_host = "localhost" 
t_port = "5432" 
t_dbname = "omar"
t_user = "admin"
t_pw = "admin"

#init classnames
m=1
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    CheckIn.append(0)
    m = m+1
print(classNames)

#encoding function
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def Attendance(name,n,id):
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    cursor = conn.cursor()
    v1 = n
    v2 = id
    v2 = v2 + 2
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    v3 = dtString
    cursor.execute("INSERT INTO hr_attendance (id, employee_id, check_in) VALUES(%s, %s, %s)", (v1, v2, v3))
    conn.commit() # <- We MUST commit to reflect the inserted data
    cursor.close()
    conn.close()

#Mark presence with date
def AttendanceOut(name,n,id):
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    cursor = conn.cursor()
    v1 = n
    v2 = id
    v2 = v2 + 2
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    v3 = dtString
    cursor.execute("UPDATE hr_attendance set check_out = %s where id = %s", (v3, v1))
    conn.commit() # <- We MUST commit to reflect the inserted data
    cursor.close()
    conn.close()
 
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
n = 5
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

 
while True:
    ret0, img = cap.read()
    ret1, img2 = cap2.read()
    if (ret0):
        # Display the resulting frame
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            R = 255
            G = 0
            if matches[matchIndex]:
                name = classNames[matchIndex]
                R = 0
                G = 255
                id = classNames.index(name)
                #print(name)
                if CheckIn[id] == 0:
                    n = n+1
                    print(name)
                    #id = 2
                    Attendance(name,n,id)
                    CheckIn[id] = n
            else: name = 'Unknown'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,G,R),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,G,R),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #print(markedname)
        cv2.imshow('Webcam',img)
    #feed 2
    if (ret1):
        imgS2 = cv2.resize(img2,(0,0),None,0.25,0.25)
        imgS2 = cv2.cvtColor(imgS2, cv2.COLOR_BGR2RGB)
 
        facesCurFrame2 = face_recognition.face_locations(imgS2)
        encodesCurFrame2 = face_recognition.face_encodings(imgS2,facesCurFrame2)
 
        for encodeFace,faceLoc in zip(encodesCurFrame2,facesCurFrame2):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)
            R = 255
            G = 0
            if matches[matchIndex]:
                name2 = classNames[matchIndex]
                R = 0
                G = 255
                id = classNames.index(name2)
                #print(name)
                if CheckIn[id] != 0:
                    #id = 2
                    AttendanceOut(name2,CheckIn[id],id)
                    CheckIn[id] = 0
                    #CheckIn[id] = False
            else: name2 = 'Unknown'
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img2,(x1,y1),(x2,y2),(0,G,R),2)
            cv2.rectangle(img2,(x1,y2-35),(x2,y2),(0,G,R),cv2.FILLED)
            cv2.putText(img2,name2,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv2.imshow('Webcam2',img2)
    cv2.waitKey(1)
cap.release()
cap2.release()
cv2.destroyAllWindows()
