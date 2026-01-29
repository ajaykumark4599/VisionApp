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

    height, width, channels = img.shape

    output_path = os.path.join(output_dir, "image_basics_output.jpg")
    cv2.imwrite(output_path, img)

    return {
        "status": "success",
        "metadata": {
            "height": height,
            "width": width,
            "channels": channels
        },
        "output": output_path
    }
