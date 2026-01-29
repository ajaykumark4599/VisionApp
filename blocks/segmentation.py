import cv2
import numpy as np
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 11: Image Segmentation
    - Otsu Thresholding
    - K-Means Clustering
    """

    img = cv2.imread(input_path)
    if img is None:
        return {
            "status": "error",
            "message": "Input image not found"
        }

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------
    # 1. Otsu Thresholding (Grayscale)
    # ----------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_path = os.path.join(output_dir, "segmentation_otsu.jpg")
    cv2.imwrite(otsu_path, otsu)

    # ----------------------------------
    # 2. K-Means Segmentation (Color)
    # ----------------------------------
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    K = 3  # number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(
        Z,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)

    kmeans_path = os.path.join(output_dir, "segmentation_kmeans.jpg")
    cv2.imwrite(kmeans_path, segmented)

    return {
        "status": "success",
        "experiment": "Exp 11 - Image Segmentation",
        "methods": ["Otsu Thresholding", "K-Means Clustering"],
        "output": {
            "otsu": otsu_path,
            "kmeans": kmeans_path
        }
    }
