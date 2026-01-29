import cv2
import os
import numpy as np

def run(input_path, output_dir, params=None):
    img = cv2.imread(input_path)

    if img is None:
        return {
            "status": "error",
            "message": "Image not found"
        }

    os.makedirs(output_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    paths = {
        "binary": os.path.join(output_dir, "binary.jpg"),
        "erosion": os.path.join(output_dir, "erosion.jpg"),
        "dilation": os.path.join(output_dir, "dilation.jpg"),
        "opening": os.path.join(output_dir, "opening.jpg"),
        "closing": os.path.join(output_dir, "closing.jpg"),
    }

    cv2.imwrite(paths["binary"], binary)
    cv2.imwrite(paths["erosion"], erosion)
    cv2.imwrite(paths["dilation"], dilation)
    cv2.imwrite(paths["opening"], opening)
    cv2.imwrite(paths["closing"], closing)

    return {
        "status": "success",
        "metadata": {
            "experiment": "Exp 5 - Morphological Operations"
        },
        "outputs": paths
    }
