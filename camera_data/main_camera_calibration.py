"""
This script calibrates the camera and stores the intrinsic camera parameters and extrinsic parameters in a file.
"""
import numpy as np
import cv2
import os

# Define the size of the checkerboard pattern
pattern_size = (8, 5)

# Define the size of the square in the checkerboard pattern (in mm)
square_size = 22.5

# Create an object to store the calibration data
calibration_data = {
    'camera_matrix': None,
    'dist_coeffs': None,
    'rvecs': None,
    'tvecs': None
}

# Define the termination criteria for the calibration process
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Initialize the list of object points and image points
obj_points = []
img_points = []

# Create the 3D object points for the checkerboard pattern
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# Start the camera
cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard pattern in the grayscale image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, add them to the list of image points and object points
    if ret == True:
        count += 1
        obj_points.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(frame, pattern_size, corners, ret)

    # Display the image
    cv2.imshow('Calibration', frame)

    # Wait for a key press
    key_pressed = cv2.waitKey(60)

    if key_pressed == 27 or count == 40:
        break

# Release the camera
cap.release()

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Store the intrinsic camera parameters and extrinsic parameters in a file
calibration_data['camera_matrix'] = camera_matrix   # Intrinsic camera matrix
calibration_data['dist_coeffs'] = dist_coeffs    # Distortion coefficients
calibration_data['rvecs'] = rvecs       # Rotation specified as a 3×1 vector  
calibration_data['tvecs'] = tvecs    # Translation specified as a 3×1 vector

# Save the calibration data in a file
output_directory = "camera_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(output_directory+"\\calibration_data.txt", "w") as f:
    for key, arr in calibration_data.items():
        f.write(key + ':\n')
        # np.savetxt(f, arr, fmt='%f')
        arr = np.array(arr)
        np.savetxt(f, arr.reshape(np.shape(arr)[0],-1), fmt='%f')

# Destroy all windows
cv2.destroyAllWindows()
