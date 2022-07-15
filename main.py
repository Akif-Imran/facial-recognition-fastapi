import shutil
import face_recognition
from fastapi import FastAPI, UploadFile, File, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field, Required
import os
import pyodbc as db
import cv2
import numpy as np


class Student(BaseModel):
    regNo: str
    firstName: str
    lastName: str
    semester: int
    degree: str
    discipline: str
    section: str


class FeatureSet(BaseModel):
    id: int | None = None
    path: str = "F:/Projects University/FeatureSet/"
    regNo: str


images = []
classNames = []
testImages = []
knownEncodingsList = []
list_of_regno = []
# students: dict = {
#     '19-ARID-0069': {
#         'firstName': 'Akif',
#         'lastName': 'Imran',
#         'semester': 7,
#         'degree': 'BS',
#         'discipline': 'CS',
#         'section': 'A',
#     },
#     '19-ARID-0074': {
#         'firstName': 'Ali',
#         'lastName': 'Murtaza',
#         'semester': 7,
#         'degree': 'BS',
#         'discipline': 'CS',
#         'section': 'A',
#     },
#     '19-ARID-0090': {
#         'firstName': 'Hammad',
#         'lastName': 'Abbasi',
#         'semester': 7,
#         'degree': 'BS',
#         'discipline': 'CS',
#         'section': 'A',
#     },
#     '19-ARID-0092': {
#         'firstName': 'Hamaza',
#         'lastName': 'Shabbir',
#         'semester': 7,
#         'degree': 'BS',
#         'discipline': 'CS',
#         'section': 'A',
#     }
# }
students: list = []
recogTestImageList = []
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
testImagePath = "TestImages"
featureImagePath = "F:/Projects University/FeatureSet"


def load_test_images():
    testImageList = os.listdir(testImagePath)
    for testImage in testImageList:
        testImageLoaded = cv2.imread(f'{testImagePath}/{testImage}')
        testImages.append(testImageLoaded)
    print("Test Images Loaded Successfully")


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def get_student_with_images():
    students.clear()
    images.clear()
    classNames.clear()
    connection = db.connect('Driver={SQL Server};'
                            'Server=DESKTOP-DFIE2SK;'
                            'Database=Fyp;'
                            'Trusted_Connection=yes;')
    cursor = connection.cursor()
    cursor.execute('select * from Student s inner join featureSet f on s.Regno = f.StudentId')
    for student in cursor:
        currentImage = cv2.imread(student[8])
        images.append(currentImage)
        classNames.append(f'{student[1]} {student[2]}')
        students.append(
            {
                'regNo': student[0],
                'firstName': student[1],
                'lastName': student[2],
                'semester': student[3],
                'degree': student[4],
                'discipline': student[5],
                'section': student[6],
            }
        )
    print(len(images))
    print(classNames)
    print()


def run_facial_recognition():
    list_of_regno.clear()
    for testImage in testImages:
        cvtTestImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)

        locationOfFacesInFrame = face_recognition.face_locations(cvtTestImage)
        encodingsOfFacesInFrame = face_recognition.face_encodings(cvtTestImage, locationOfFacesInFrame)

        for locationOfFace, encodingOfFace, val in zip(locationOfFacesInFrame, encodingsOfFacesInFrame,
                                                       range(len(locationOfFacesInFrame))):
            matches = face_recognition.compare_faces(knownEncodingsList, encodingOfFace, .54)
            distance = face_recognition.face_distance(knownEncodingsList, encodingOfFace)
            matchIndex = np.argmin(distance)

            if matches[matchIndex]:
                name = classNames[matchIndex]
                list_of_regno.append(students[matchIndex]['regNo'])
                top, right, bottom, left = locationOfFace
                # Draw a box around the face
                cv2.rectangle(testImage, (left, top), (right, bottom), (0, 0, 255), 5)
                cv2.putText(testImage, name, (left, (top - (top - bottom) - 10)), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 2)

        recogTestImageList.append(testImage)
        # cv2.imshow(f"Recognition-{val}", cv2.resize(testImage, (960, 540)))


get_student_with_images()
knownEncodingsList = find_encodings(images)
print("Encoding Complete")


def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Fyp FastAPI"}


@app.post("/uploadImage/")
async def uploadImages(background_task: BackgroundTasks,
                       files: list[UploadFile] =
                       File(default=Required, title="Upload Image File")):
    for file in files:
        if allowed_file(file.filename):
            fp = open(f'TestImages/{file.filename}', 'w')
            fp.close()
            with open(f'TestImages/{file.filename}', 'wb+') as buffer:
                shutil.copyfileobj(file.file, buffer)
    background_task.add_task(load_test_images)
    return {"message": f'file were uploaded successfully.'}


@app.get("/recognizeFaces/")
async def recognition():
    run_facial_recognition()
    return {"message": "Facial Recognition Completed."}


@app.get("/attendance-regno/")
async def get_student_list():
    return list_of_regno;


@app.post("/student/save/")
async def save_student(s: Student, image: UploadFile = File(default=Required, title="Student frontal face Image.")):
    try:
        fullPath = f'{featureImagePath}/{s.firstName} {s.lastName}.jpg'
        print(image.filename)
        if allowed_file(image.filename):
            fp = open(fullPath, 'w')
            fp.close()
            with open(fullPath, 'wb+') as buffer:
                shutil.copyfileobj(image.file, buffer)
        try:
            currentImage = cv2.imread(fullPath)
            cvtCurrentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2RGB)
            faceLocation = face_recognition.face_locations(cvtCurrentImage)
            if len(faceLocation) == 0:
                return {"Error": 'No faces Detected. Please select a good resolution image with a face present.'}
            faceEncode = face_recognition.face_encodings(cvtCurrentImage, faceLocation)
            images.append(currentImage)
            classNames.append(f'{s.firstName} {s.lastName}')
            knownEncodingsList.append(faceEncode)
        except Exception as e:
            return {"Error": f"{str(e)}"}
        else:
            students.append({
                'regNo': s.regNo,
                'firstName': s.firstName,
                'lastName': s.lastName,
                'semester': s.semester,
                'degree': s.degree,
                'discipline': s.discipline,
                'section': s.section,
            })
            connection = db.connect('Driver={SQL Server};'
                                    'Server=DESKTOP-DFIE2SK;'
                                    'Database=Fyp;'
                                    'Trusted_Connection=yes;')
            cursor = connection.cursor()
            cursor.execute(
                'insert into Student (Regno,FirstName,LastName,Semester,Degree,Discipline,Section) values(?,?,?,?,?,?,?)',
                s.regNo, s.firstName, s.lastName, s.semester, s.degree, s.discipline, s.section)
            cursor.commit()
            cursor = connection.cursor()
            cursor.execute('insert into featureSet (Path,StudentId) values(?,?)', fullPath, s.regNo)
            cursor.commit()
            connection.close()
    except Exception as e:
        return {'Error': f"{str(e)}."}
    else:
        return {'message': 'Student Added Successfully.'}
