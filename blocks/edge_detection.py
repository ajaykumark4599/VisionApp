import cv2
import os

def run(input_path, output_dir, params=None):
    if params is None:
        params = {}

    low = int(params.get("low", 100))
    high = int(params.get("high", 200))

    img = cv2.imread(input_path)

    if img is None:
        return {
            "status": "error",
            "message": "Image not found"
        }

    os.makedirs(output_dir, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)

    output_path = os.path.join(output_dir, f"edges_{low}_{high}.jpg")
    cv2.imwrite(output_path, edges)

    return {
        "status": "success",
        "metadata": {
            "method": "Canny",
            "low_threshold": low,
            "high_threshold": high
        },
        "output": output_path
    }

