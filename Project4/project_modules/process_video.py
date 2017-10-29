from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import matplotlib.pyplot as plt


def process_video(video_path_in, video_path_out, process_image_fn):
	clip_in = VideoFileClip(video_path_in)
	clip_out = clip_in.fl_image(process_image_fn) 
	clip_out.write_videofile(video_path_out, audio=False)


# Draw the found lane overlay onto the image
def draw_lane_image(input_image, binary_img, left_fitx, right_fitx, Minv):   
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, input_image.shape[0]-1, input_image.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,0), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,0,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_LINEAR) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(input_image, 1, newwarp, 0.3, 0)
    return result

# Output the lane radius of curvature & center offset onto the image
def output_curvature_data(input_image, left_curverad, car_position, left_line): 
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    text_color = (255, 255, 255)
    line_type = 2

    # Median of last N
    left_line.curve_list = left_line.curve_list[-5:]
    curvature = np.median(left_line.curve_list) 

    text_left = 'Curvature : {} m'.format(int(curvature))

    if car_position < 0.0:
        text_center = 'Position : {0:.1f} m left of center'.format(round(-car_position, 1))
    else:
        text_center = 'Position : {0:.1f} m right of center'.format(round(car_position, 1))

    position_left = (40, 60)
    position_center = (40, 90)

    output_image = cv2.putText(input_image,text_left, position_left, font, font_scale, text_color, line_type)
    output_image = cv2.putText(output_image,text_center, position_center, font, font_scale, text_color, line_type)

    return output_image


