import cv2
import os

def run(input_path, output_dir, params=None):
    img = cv2.imread(input_path)

    if img is None:
        return {
            "status": "error",
            "message": "Image not found"
        }

    os.makedirs(output_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gray_path = os.path.join(output_dir, "gray.jpg")
    hsv_path = os.path.join(output_dir, "hsv.jpg")

    cv2.imwrite(gray_path, gray)
    cv2.imwrite(hsv_path, hsv)

    return {
        "status": "success",
        "metadata": {
            "experiment": "Exp 2 - Color Space Conversion"
        },
        "outputs": {
            "grayscale": gray_path,
            "hsv": hsv_path
        }
    }
