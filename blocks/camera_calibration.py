import cv2
import numpy as np
import os
import glob

def run(input_dir, output_dir, params=None):
    os.makedirs(output_dir, exist_ok=True)

    # INNER corners (adjust if needed)
    chessboard_size = (9,6)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(input_dir, "*.jpg"))

    if len(images) == 0:
        return {"status": "error", "message": "No calibration images found"}

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) < 5:
        return {
            "status": "error",
            "message": f"Only {len(objpoints)} valid detections. Need at least 5."
        }

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    calib_path = os.path.join(output_dir, "camera_calibration.npz")
    np.savez(calib_path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs)

    return {
        "status": "success",
        "experiment": "Exp 6 - Camera Calibration",
        "images_used": len(objpoints),
        "reprojection_error": ret,
        "output": calib_path
    }
