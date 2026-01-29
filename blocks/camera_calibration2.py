import cv2
import numpy as np
import os
import glob

def run(input_dir, output_dir, params=None):
    """
    Experiment 6: Camera Calibration
    - Automatically detects chessboard inner-corner size
    - Uses the size with maximum valid detections
    - Saves calibration as .npz
    """

    if not os.path.isdir(input_dir):
        return {"status": "error", "message": "Input directory not found"}

    os.makedirs(output_dir, exist_ok=True)

    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    if len(images) == 0:
        return {"status": "error", "message": "No calibration images found"}

    # --------------------------------------------------
    # Generate flexible search space for inner corners
    # --------------------------------------------------
    possible_sizes = []
    for cols in range(4, 13):      # horizontal inner corners
        for rows in range(4, 10):  # vertical inner corners
            if cols * rows >= 20:  # ignore tiny boards
                possible_sizes.append((cols, rows))

    best_size = None
    best_count = 0
    best_objpoints = None
    best_imgpoints = None
    final_gray = None

    # --------------------------------------------------
    # Try all possible chessboard sizes
    # --------------------------------------------------
    for size in possible_sizes:
        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray,
                size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                final_gray = gray

        if len(objpoints) > best_count:
            best_count = len(objpoints)
            best_size = size
            best_objpoints = objpoints
            best_imgpoints = imgpoints

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    if best_size is None or best_count < 5:
        return {
            "status": "error",
            "message": "No valid chessboard pattern detected reliably"
        }

    # --------------------------------------------------
    # Camera Calibration
    # --------------------------------------------------
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        best_objpoints,
        best_imgpoints,
        final_gray.shape[::-1],
        None,
        None
    )

    calib_path = os.path.join(output_dir, "camera_calibration2.npz")
    np.savez(
        calib_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        chessboard_size=best_size
    )

    return {
        "status": "success",
        "experiment": "Exp 6 - Camera Calibration (Free Size)",
        "detected_chessboard_size": best_size,
        "images_used": best_count,
        "reprojection_error": ret,
        "output": calib_path
    }
