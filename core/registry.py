from blocks import (
    image_basics,
    color_space,
    filtering,
    edge_detection,
    morphology,
    camera_calibration,
    undistortion,
    perspective_transform,
    feature_matching,
    optical_flow,
    segmentation,
    object_detection,
    face_recognition,
    object_tracking
)

BLOCKS = {
    "1": ("Image Basics", image_basics.run),
    "2": ("Color Space Conversion", color_space.run),
    "3": ("Filtering", filtering.run),
    "4": ("Edge Detection", edge_detection.run),
    "5": ("Morphology", morphology.run),
    "6": ("Camera Calibration", camera_calibration.run),
    "7": ("Undistortion", undistortion.run),
    "8": ("Perspective Transform", perspective_transform.run),
    "9": ("Feature Matching", feature_matching.run),
    "10": ("Optical Flow", optical_flow.run),
    "11": ("Segmentation", segmentation.run),
    "12": ("Object Detection", object_detection.run),
    "13": ("Face Recognition", face_recognition.run),
    "14": ("Object Tracking", object_tracking.run),
}
