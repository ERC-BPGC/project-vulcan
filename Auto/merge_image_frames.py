import cv2
import numpy as np

# Initialize camera capture for two camera streams
cap1 = cv2.VideoCapture(0)  # Adjust the camera index as needed
cap2 = cv2.VideoCapture(2)  # Adjust the camera index as needed

# Check if the camera streams are opened successfully
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Camera not found")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Couldn't read frames from the cameras")
        break

    # Resize the frames to have the same dimensions
    frame1 = cv2.resize(frame1, (640, 480))  # Adjust the dimensions as needed
    frame2 = cv2.resize(frame2, (640, 480))  # Adjust the dimensions as needed

    # Detect and compute key points and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Match key points
    bf = cv2.BFMatcher()
    
    # Convert descriptors to np.float32
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    # Match key points
    matches = bf.knnMatch(des1, des2, k=2)

    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the second image using the homography matrix
        stitched_frame = cv2.warpPerspective(frame2, M, (frame1.shape[1] + frame2.shape[1], frame1.shape[0]))

        # Blend the two frames
        stitched_frame[0:frame1.shape[0], 0:frame1.shape[1]] = frame1
    else:
        stitched_frame = frame1

    # Display the stitched frame
    cv2.imshow('Stitched Frames', stitched_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera streams and close the OpenCV window
cap1.release()
cap2.release()
cv2.destroyAllWindows()
