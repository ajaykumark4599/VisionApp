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

    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # Median Filter
    median = cv2.medianBlur(img, 5)

    gaussian_path = os.path.join(output_dir, "gaussian_blur.jpg")
    median_path = os.path.join(output_dir, "median_filter.jpg")

    cv2.imwrite(gaussian_path, gaussian)
    cv2.imwrite(median_path, median)

    return {
        "status": "success",
        "metadata": {
            "experiment": "Exp 3 - Noise Removal & Filtering",
            "filters": ["Gaussian Blur", "Median Filter"]
        },
        "outputs": {
            "gaussian": gaussian_path,
            "median": median_path
        }
    }
