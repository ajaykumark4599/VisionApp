import cv2
import os

def main():
    save_dir = "data/calibration_images"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible")
        return

    count = 0
    print("Press 'c' to capture image, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Calibration Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_path = os.path.join(save_dir, f"calib_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured {img_path}")
            count += 1

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
