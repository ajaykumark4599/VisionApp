import cv2
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 12: Object Detection using Haar Cascade (Face Detection)
    """

    cascade_path = "models/haarcascade_frontalface.xml"

    if not os.path.exists(cascade_path):
        return {
            "status": "error",
            "message": "Haar cascade file not found"
        }

    img = cv2.imread(input_path)
    if img is None:
        return {
            "status": "error",
            "message": "Input image not found"
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "object_detection.jpg")
    cv2.imwrite(output_path, img)

    return {
        "status": "success",
        "experiment": "Exp 12 - Object Detection (Haar Cascade)",
        "objects_detected": len(faces),
        "output": output_path
    }
