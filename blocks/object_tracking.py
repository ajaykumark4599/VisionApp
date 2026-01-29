import cv2
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 14: Object Tracking using CSRT Tracker
    """

    if params is None or "second_image" not in params:
        return {
            "status": "error",
            "message": "Second image path not provided"
        }

    img1 = cv2.imread(input_path)
    img2 = cv2.imread(params["second_image"])

    if img1 is None or img2 is None:
        return {
            "status": "error",
            "message": "One or both images not found"
        }

    cascade_path = "models/haarcascade_frontalface.xml"
    if not os.path.exists(cascade_path):
        return {
            "status": "error",
            "message": "Haar cascade not found"
        }

    face_cascade = cv2.CascadeClassifier(cascade_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray1, 1.2, 5)

    if len(faces) == 0:
        return {
            "status": "error",
            "message": "No object detected to initialize tracker"
        }

    # Initialize tracker with first detected face
    x, y, w, h = faces[0]
    tracker = cv2.TrackerCSRT_create()
    tracker.init(img1, (x, y, w, h))

    success, bbox = tracker.update(img2)

    if not success:
        return {
            "status": "error",
            "message": "Tracking failed"
        }

    x, y, w, h = map(int, bbox)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "object_tracking.jpg")
    cv2.imwrite(output_path, img2)

    return {
        "status": "success",
        "experiment": "Exp 14 - Object Tracking (CSRT)",
        "output": output_path
    }
