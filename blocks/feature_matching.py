import cv2
import numpy as np
import os

def run(input_path, output_dir, params=None):
    """
    Experiment 9: Feature Detection and Matching (ORB)
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

    # ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return {
            "status": "error",
            "message": "Could not compute descriptors"
        }

    # Brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "feature_matches.jpg")
    cv2.imwrite(output_path, matched_img)

    return {
        "status": "success",
        "experiment": "Exp 9 - Feature Detection and Matching (ORB)",
        "num_keypoints_img1": len(kp1),
        "num_keypoints_img2": len(kp2),
        "num_matches": len(matches),
        "output": output_path
    }
