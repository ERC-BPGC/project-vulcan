import cv2

def list_available_cameras():
    available_cameras = []

    for i in range(10):  # You can adjust the range based on the number of cameras you expect to find
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()

    return available_cameras

if __name__ == "__main__":
    cameras = list_available_cameras()

    if cameras:
        print("Available Cameras:")
        for camera in cameras:
            print(camera)
    else:
        print("No cameras found.")
