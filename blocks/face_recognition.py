import cv2
import os
import numpy as np

def run(input_path, output_dir, params=None):
    """
    Experiment 13: Face Recognition using LBPH
    """

    faces_dir = "data/faces"
    cascade_path = "models/haarcascade_frontalface.xml"

    if not os.path.exists(faces_dir):
        return {"status": "error", "message": "Faces dataset not found"}

    if not os.path.exists(cascade_path):
        return {"status": "error", "message": "Haar cascade not found"}

    face_cascade = cv2.CascadeClassifier(cascade_path)

    labels = {}
    label_id = 0
    training_faces = []
    training_labels = []

    # ---------------------------------
    # Load training faces
    # ---------------------------------
    for person in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person)
        if not os.path.isdir(person_path):
            continue

        labels[label_id] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            training_faces.append(img)
            training_labels.append(label_id)

        label_id += 1

    if len(training_faces) == 0:
        return {"status": "error", "message": "No training images found"}

    # ---------------------------------
    # Train LBPH recognizer
    # ---------------------------------
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(training_faces, np.array(training_labels))

    # ---------------------------------
    # Recognize face in input image
    # ---------------------------------
    img = cv2.imread(input_path)
    if img is None:
        return {"status": "error", "message": "Input image not found"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)

        name = labels[label]
        text = f"{name} ({int(confidence)})"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "face_recognition.jpg")
    cv2.imwrite(output_path, img)

    return {
        "status": "success",
        "experiment": "Exp 13 - Face Recognition (LBPH)",
        "faces_detected": len(faces),
        "output": output_path
    }
