import cv2
import numpy as np
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 7: Image Undistortion
    - Uses camera_calibration.npz from Exp 6
    - Removes lens distortion
    """

    calib_file = os.path.join(output_dir, "camera_calibration.npz")

    if not os.path.exists(calib_file):
        return {
            "status": "error",
            "message": "Calibration file not found. Run Exp 6 first."
        }

    img = cv2.imread(input_path)
    if img is None:
        return {
            "status": "error",
            "message": "Input image not found"
        }

    # Load calibration data
    data = np.load(calib_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    h, w = img.shape[:2]

    # Get optimal camera matrix
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    # Undistort image
    undistorted = cv2.undistort(
        img,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_mtx
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "undistorted.jpg")
    cv2.imwrite(output_path, undistorted)

    return {
        "status": "success",
        "experiment": "Exp 7 - Image Undistortion",
        "output": output_path
    }
