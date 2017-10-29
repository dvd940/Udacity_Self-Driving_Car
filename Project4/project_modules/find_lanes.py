import numpy as np
import cv2
import matplotlib.pyplot as plt


# FInd lanes in an image
def find_lanes(input_image):    
    histogram = np.sum(input_image[input_image.shape[0]//2:,:], axis=0)  # Histogram of the bottom half of the image    
    out_img = np.dstack((input_image, input_image, input_image))*255  # Output image to visualize the result

    # Find the peak of the left and right halves of the histogram as starting point for the lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    
    nwindows = 9 # number of sliding windows    
    window_height = np.int(input_image.shape[0]/nwindows)  # height of windows
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = input_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100  # Set the width of the windows +/- margin
    minpix = 50 # Set minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = input_image.shape[0] - (window+1)*window_height
        win_y_high = input_image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds


def curvature_measurement(input_image, left_fit, right_fit, left_lane_inds, right_lane_inds):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty = np.linspace(0, input_image.shape[0]-1, input_image.shape[0] )
    y_eval = np.max(ploty)

    leftx= left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx= right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_curverad, right_curverad, left_fitx, right_fitx


# Find car position in the lane
def vehicle_position(input_image, left_fit, right_fit):
    xm_per_pix = 3.7/700
    car_center = input_image.shape[1]/2
    ploty = np.linspace(0, input_image.shape[0]-1, input_image.shape[0] )
    y_max = np.max(ploty)

    leftx_bottom = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    rightx_bottom = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    lane_center = (leftx_bottom + rightx_bottom)/2
    car_position = (lane_center - car_center) * xm_per_pix
    return car_position


# Rudimentary line smoothing by averaging over N fits
def line_smoothing(lane, fitx, curvature):
   
    lane.recent_xfitted.append(fitx)
    smoothing_factor = 10
    if len(lane.recent_xfitted) > smoothing_factor:
        lane.recent_xfitted.pop(0)

    lane.bestx = np.median(lane.recent_xfitted,axis=0)
    
    return lane.bestx if lane.bestx is not None else fitx


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])] 
        # Keep a list of the curvature for display
        self.curve_list = []


