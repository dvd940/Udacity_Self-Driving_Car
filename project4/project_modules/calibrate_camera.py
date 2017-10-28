import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Read calibration images
def get_calibration_image_names():
    images = glob.glob('camera_cal/calibration*.jpg')
    return images

# Calibrate the camera
def calibrate_camera(images, visualize=False):
    # Arrays to store object and image points for all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Preapre object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)  # x, y coordinates

    if visualize == True:
        # params for displaying images. 
        plt.figure(figsize=(15, 10))
        num_of_samples = len(images)  # How many samples to display
        cols = 4  # How many columns in display
        num_rows = num_of_samples / cols
        image_index = 0

    for image in images:
        img = mpimg.imread(image)
        
        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        
        #if corners found, add to object points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)            
            
            if visualize == True:
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                plt.subplot(num_rows, cols, image_index+1)
                plt.suptitle('Checkerboard with found corners', fontsize= 30)
                plt.axis('off')
                plt.imshow(img)
                image_index += 1
   
    return(imgpoints, objpoints)



