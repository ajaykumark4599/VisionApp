import cv2
import numpy as np
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 8: Perspective Transformation (Homography)
    """

    if params is None:
        return {
            "status": "error",
            "message": "Source and destination points not provided"
        }

    img = cv2.imread(input_path)
    if img is None:
        return {
            "status": "error",
            "message": "Input image not found"
        }

    try:
        src_pts = np.array(params["src_points"], dtype=np.float32)
        dst_pts = np.array(params["dst_points"], dtype=np.float32)
    except KeyError:
        return {
            "status": "error",
            "message": "src_points or dst_points missing in params"
        }

    if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
        return {
            "status": "error",
            "message": "Exactly 4 source and 4 destination points required"
        }

    # Compute homography
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    width = int(max(dst_pts[:, 0]))
    height = int(max(dst_pts[:, 1]))

    warped = cv2.warpPerspective(img, H, (width, height))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "perspective_warped.jpg")
    cv2.imwrite(output_path, warped)

    return {
        "status": "success",
        "experiment": "Exp 8 - Perspective Transformation",
        "output": output_path
    }
