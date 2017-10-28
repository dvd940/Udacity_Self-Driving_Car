import cv2
import numpy as np


# Undistort image
def undistort_image(img, imgpoints, objpoints):    
    # Convert to gray scale
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# Apply bionary color threshold to image
def color_threshold(img):

	R = img[:,:,0]
	G = img[:,:,1]

	thresh = (220, 255)
	r_binary = np.zeros_like(R)
	r_binary[(R > thresh[0]) & (R <= thresh[1])] = 1


	thresh = (220, 235)
	g_binary = np.zeros_like(G)
	g_binary[(G > thresh[0]) & (G <= thresh[1])] = 1

	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

	L = hls[:,:,1]
	S = hls[:,:,2]

	thresh = (220, 255)
	s_binary = np.zeros_like(S)
	s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1


	thresh = (200, 210)
	l_binary = np.zeros_like(L)
	l_binary[(L > thresh[0]) & (L <= thresh[1])] = 1

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(s_binary)
	combined_binary[((s_binary == 1) & (l_binary == 1) | (r_binary == 1) | (g_binary == 1) )] = 1

	return combined_binary



# Perspective Transform
def perspective_transform(img, return_pts=False):

    src = np.float32([[275,675],
                     [580,460],
                     [705,460],
                     [1045,675]])

    dst = np.float32([[275,675],
                     [275,0],
                     [1045,0],
                     [1045,675]])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)  # for projecting back onto original image

    img_size = img.shape[:2][::-1]
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    if return_pts == True:
    	return warped_img, Minv, src, dst
    else:
    	return warped_img, Minv