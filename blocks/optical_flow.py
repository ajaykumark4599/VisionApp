import cv2
import numpy as np
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 10: Optical Flow using Lucas-Kanade method
    """

    if params is None or "second_image" not in params:
        return {
            "status": "error",
            "message": "Second image path not provided"
        }

    img1 = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(params["second_image"], cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return {
            "status": "error",
            "message": "One or both images not found"
        }

    # Detect good features to track
    p0 = cv2.goodFeaturesToTrack(
        img1,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7
    )

    if p0 is None:
        return {
            "status": "error",
            "message": "No features detected to track"
        }

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img1,
        img2,
        p0,
        None
    )

    # Select good points
    good_old = p0[st == 1]
    good_new = p1[st == 1]

    # Convert first image to color for visualization
    vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for old, new in zip(good_old, good_new):
        x_old, y_old = old.ravel()
        x_new, y_new = new.ravel()

        cv2.arrowedLine(
            vis,
            (int(x_old), int(y_old)),
            (int(x_new), int(y_new)),
            (0, 255, 0),
            1,
            tipLength=0.3
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "optical_flow.jpg")
    cv2.imwrite(output_path, vis)

    return {
        "status": "success",
        "experiment": "Exp 10 - Optical Flow (Lucas-Kanade)",
        "tracked_points": len(good_new),
        "output": output_path
    }
