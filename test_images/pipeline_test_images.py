import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob as glb
import os
import cv2
import pickle
import sys
#import numpy

np.set_printoptions(threshold=sys.maxsize)


class Line():
    def __init__(self,n=0):
        # number of fits
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = list()
        #polynomial coefficients for the most recent fit
        self.current_fit = None # [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def update_line(self,detected,recent_xfit,curr_fit,new_radius,new_base_dist,new_x_vals,y_vals):
        self.n += 1
        self.detected = detected
        self.recent_xfitted.append(recent_xfit)
        self.bestx=np.average(self.recent_xfitted,axis=0)
        #print('self.best_fit',self.best_fit)
        if len( self.best_fit) ==0:
            self.best_fit=curr_fit
        else:
            self.best_fit =np.average([self.best_fit,curr_fit],axis=0)
        self.current_fit = curr_fit
        self.radius_of_curvature = new_radius
        self.line_base_pos = new_base_dist
        #self.diffs = np.array([0, 0, 0], dtype='float') [[n2 - n1 for n1, n2 in zip(l1, l2)] for l1, l2 in zip(a, b)]
        self.allx = new_x_vals
        self.ally = y_vals
#################################################################################################################
def create_new_folder (new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return
#################################################################################################################

def generate_new_fpath (fname,saveas_dir,key_word):
    base = os.path.basename(fname)
    new_fname = os.path.splitext(base)[0] + key_word + os.path.splitext(base)[1]
    new_fpath = saveas_dir + new_fname

    return new_fpath
#################################################################################################################

def save_img_as (orig_img_name,save_as_path,keyword,image_to_save):
    fpath = generate_new_fpath(orig_img_name, save_as_path, keyword)
    mpimg.imsave(fpath, image_to_save)

    return
#################################################################################################################
def load_mtx_dist(fpath):
    # load mtx and dist db

    db_file = open(fpath, 'rb')
    pickle_dist = pickle.load(db_file)
    mtx = pickle_dist['mtx']
    dist = pickle_dist['dist']

    return mtx, dist

#################################################################################################################
def load_mtx_transform(fpath):
    # load transf. mtx

    db_file = open(fpath, 'rb')
    pickle_dist = pickle.load(db_file)
    mtx= pickle_dist['TM']
    mtx_inverse=pickle_dist['TMinv']
    return mtx,mtx_inverse

#################################################################################################################
def undist (img,mtx,dist,fname,saveas_dir):

    #create undistorted image
    undist_image=cv2.undistort(img,mtx,dist,None,mtx)
    #mpimg.imsave(new_fpath, undist_image)
    return undist_image

#################################################################################################################

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient=='x':
        dx=1
        dy=0
    elif orient=='y':
        dx=0
        dy=1
    else :
        dx=1
        dy=0
    sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy,ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary=np.zeros_like(scaled_sobel)
    # Apply threshold
    grad_binary[(scaled_sobel<=thresh[1])&(scaled_sobel>=thresh[0])]=1

    return grad_binary
#################################################################################################################

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobel_x=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=sobel_kernel)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    scaled_mag=np.uint8(255*grad_mag/(np.max(grad_mag)))

    mag_binary=np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag<=mag_thresh[1])&(scaled_mag>=mag_thresh[0])]=1
    # Apply threshold
    return mag_binary
#################################################################################################################

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel=np.arctan2(np.absolute(sobel_y),np.absolute(sobel_x))

    dir_binary=np.zeros_like(dir_sobel)
    # Apply threshold
    dir_binary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
    return dir_binary
#################################################################################################################
def get_horizontal_pixels_density_histogram_of_thresh_img(img,bin_img,thresh_min,thresh_max):
    bin_img[(img >= thresh_min) & (img <= thresh_max)] = 1
    hist= np.sum(bin_img, axis=1)
    return hist
#################################################################################################################
def find_lanes(img, col_space, channel_no, thresh_min, thresh_max, y_to_check, min_pixels, max_thick_allowed,increment_flag=0):
    #giving an image, this function will extract a channel from a specified color space, apply threshold
    #vary the minimum threshold to achieve high content at top of the image where the lines are harder to catch

    # min_pixels:  minimum number of pixels to be found at y=0.05*ymax which is at the top picture
    # max_thick_allowed: maximum pixel numbers for a specified y position
    # y_to_check check if we have line content at the top the picture or any other position
    #increment_flag: 0 to decrement 1 to increment
    conv_img = cv2.cvtColor(img, col_space)
    channel = conv_img[:, :, channel_no]
    bin_img = np.zeros_like(channel)

    bin_img[(channel >= thresh_min) & (channel <= thresh_max)] = 1

    #get histogram of the image as x=f(y) this will let us now if we have pixels at every y position
    hist = np.sum(bin_img, axis=1)
    #check if we have pixels at a specified y position (in our case we will be looking at the top of the image
    previous_density = hist[y_to_check]

    #if we don't catch at least min_pixels at this y position, the lane is probably not detected at this level
    #if for any y position we have a horizontal sequence of pixels detected which is longer than max_thick_allowed
    #it means it is unlikely a lane that is being detected and we are going in the wrong direction
    while  (np.argmax(hist) > max_thick_allowed):
        if increment_flag==0:
            thresh_min -= 1
        else:
            thresh_min += 1
        hist = get_horizontal_pixels_density_histogram_of_thresh_img(channel, bin_img, thresh_min, thresh_max)
        print("first loop",thresh_min,np.argmax(hist))
    while (hist[y_to_check] < min_pixels)& (np.argmax(hist) < max_thick_allowed):
        if increment_flag==0:
            thresh_min -= 1
        else:
            thresh_min += 1
        hist = get_horizontal_pixels_density_histogram_of_thresh_img(channel, bin_img, thresh_min, thresh_max)

    return bin_img


#################################################################################################################
def get_binary_image_from_channel(input_img,color_space,channel_no,thresh=(0,255)):
    #converts an input image (supposed to be RGB image) to a specific color space and pick up a chosen channel from it
    #apply thresholds to obtain a binary output image
    conv_img=cv2.cvtColor(input_img, color_space)
    channel=conv_img[:,:,channel_no]
    bin_out=np.zeros_like(channel)
    bin_out[(channel>thresh[0])&(channel<thresh[1])]=1
    return bin_out

#################################################################################################################
def apply_gradient_and_colors_thresholds(input_img, th_white, th_yellow, kernel_size, th_x, th_y,th_mag, th_dir):
    # apply combined colors and gradients thresholds to obtain a binary image with two lanes as the most clear objects
    ########################################
    # detecting White lane lines
    bin_white = get_binary_image_from_channel(input_img, cv2.COLOR_RGB2LUV, 0, th_white)
    ########################################
    # detecting yellow lane lines
    bin_yellow = get_binary_image_from_channel(input_img, cv2.COLOR_RGB2LAB, 2, th_yellow)
    ########################################
    # converting to gray
    gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    # gradient thresholding
    # x gradient
    out_x = abs_sobel_thresh(gray, 'x', kernel_size, th_x)
    # y gradient
    out_y = abs_sobel_thresh(gray, 'y', kernel_size, th_y)
    # magnitude gradient
    out_mag = mag_thresh(gray, kernel_size, th_mag) # not used
    # direction gradient
    out_dir = dir_threshold(gray, kernel_size, th_dir)
    ########################################
    # combining gradients and colors thresholds
    combined_bin = np.zeros_like(gray)
    combined_bin[((out_x == 1) & (out_y == 1)) | ((out_mag==1 )&(out_dir==1) )| (bin_yellow == 1) | (bin_white == 1)] = 1

    return combined_bin

#################################################################################################################
def finding_lanes_starting_x_position(input_img):
    #locates lanes (vertical shapes) in a binary!!! image
    #get image y size
    img_y_size=input_img.shape[0]
    #Grab the bottom half of the image
    bottom_half=input_img[(img_y_size//3)*2:,:]
    #sum across pixels vertically (spot vertical shapes == lines in our case)
    hist=np.sum(bottom_half,axis=0)

    #output_img=np.dstack((input_img,input_img,input_img))*255
    mid_pt=int(hist.shape[0]//2)
    left_pos=np.argmax(hist[:mid_pt])
    right_pos=np.argmax(hist[mid_pt:])+mid_pt

    return left_pos,right_pos
#################################################################################################################
def find_lane_pixels(img, nwindows=9, margin=100, minpix=50):
    # nwindows: number of sliding windows
    # margin: margin around center point (search area)
    # min pix: minimum number of pixels to recenter
    output_img=np.dstack((img,img,img))*255
    # locate lanes
    left_lane_pos, right_lane_pos = finding_lanes_starting_x_position(img)
    delta_lanes=right_lane_pos-left_lane_pos
    cv2.circle(output_img, (left_lane_pos, bin_result_combined.shape[0]-5), radius=0, color=(0, 0, 255), thickness=20)
    cv2.circle(output_img, (right_lane_pos, bin_result_combined.shape[0] - 5), radius=0, color=(0, 255, 0),thickness=20)
    ysize = img.shape[0]
    #print (img.shape)
    # calculate height of windows based on image size and windows number
    windows_height = int(ysize // nwindows)
    # identify all active pixels
    nonzero = img.nonzero()
    #print ('nonzero',nonzero)
    nonzeroy = np.array(nonzero[0])
    #print('nonzerox',nonzeroy)
    #print ('====================================================================================================')
    nonzerox = np.array(nonzero[1])

    #print('nonzeroy',nonzerox)
    # current position for each window
    leftx_current = left_lane_pos
    rightx_current = right_lane_pos

    left_lane_inds = list()
    right_lane_inds = list()
    lane_flag=0 #incremented by
    left_turn=0
    right_turn=0
    # -1 if rectangles are recentered to left compared to last rectangle,
    #  1 if to the right

    left_flag = True  # True means pixels were found in the previous rectangle
    right_flag = True  # True means pixels were found in the previous rectangle
    iter=0
    for window in range(nwindows):

        #print('iteration No:',iter,nwindows,window)
        iter+=1

        # define windows boundaries
        win_y_top = ysize - (window + 1) * windows_height
        win_y_bottom = ysize - window * windows_height

        margin_applied=margin
        if iter<=2: margin_applied=int(margin/2)
        win_x_left_low = leftx_current - margin_applied
        win_x_left_high = leftx_current + margin_applied
        win_x_right_low = rightx_current - margin_applied
        win_x_right_high = rightx_current + margin_applied

        pt_left_low = (win_x_left_low, win_y_bottom)
        pt_left_high = (win_x_left_high, win_y_top)

        pt_right_low = (win_x_right_low, win_y_bottom)
        pt_right_high = (win_x_right_high, win_y_top)
        # draw window
        #print (pt_left_low,pt_left_high)
        cv2.rectangle(output_img, pt_left_low, pt_left_high, (0, 255, 0), 2)
        cv2.rectangle(output_img, pt_right_low, pt_right_high, (0, 255, 0), 2)

        # identify non zero (active) pixels within search area (rectangle)
        nonzero_left_inds = list(((nonzeroy <= win_y_bottom) & (nonzeroy > win_y_top) & (nonzerox >= win_x_left_low) & (
                    nonzerox <= win_x_left_high)).nonzero()[0])
        #print('nonzero_left_inds',nonzero_left_inds)
        nonzero_right_inds = list(((nonzeroy <= win_y_bottom) & (nonzeroy > win_y_top) & (nonzerox >= win_x_right_low) & (
                    nonzerox <= win_x_right_high)).nonzero()[0])

        # if not left_flag: nonzero_left_inds=list(((nonzeroy <= win_y_bottom) & (nonzeroy > win_y_top) & (nonzerox+delat_lanes >= win_x_right_low) & (
        #             nonzerox+delat_lanes <= win_x_right_high)).nonzero()[0])
        # if not right_flag:nonzero_right_inds=list(((nonzeroy <= win_y_bottom) & (nonzeroy > win_y_top) & (nonzerox-delat_lanes >= win_x_left_low) & (
        #             nonzerox-delat_lanes <= win_x_left_high)).nonzero()[0])
        left_flag = True  # True means pixels were found in the previous rectangle
        right_flag = True  # True means pixels were found in the previous rectangle
        #print('nonzero_right_inds', nonzero_right_inds)
        # append to list for each lane
        #print (type(left_lane_inds),type(nonzero_left_inds))
        left_lane_inds.append(nonzero_left_inds)
        right_lane_inds.append(nonzero_right_inds)
        # recenter if min pix is achieved
        leftx_previous = leftx_current
        rightx_previous = rightx_current

        if (len(nonzero_left_inds) >= minpix) :
            leftx_current = int(np.mean(nonzerox[nonzero_left_inds]))
            delta_left_current = leftx_current - leftx_previous
            delta_left_absolute = leftx_current - left_lane_pos
            #if (delta_left_absolute*delta_left_current<0 ) and (iter>3): leftx_current=leftx_previous+delta_left/2
        if (len(nonzero_right_inds) >= minpix) :
            rightx_current = int(np.mean(nonzerox[nonzero_right_inds]))
            delta_right_current = rightx_current - rightx_previous
            delta_right_absolute = rightx_current - right_lane_pos
           # if (delta_right_absolute * delta_right_current < 0) and (iter>3): rightx_current = rightx_previous+delta_right/2

        delta_right = rightx_current - rightx_previous
        delta_left = leftx_current - leftx_previous
        if iter >2 :
            if delta_left > 0:
                left_turn += 1
            elif delta_left < 0:
                left_turn -= 1

            if delta_right > 0:
                right_turn += 1
            elif delta_right < 0:
                right_turn -= 1

        if (len(nonzero_left_inds) < minpix)  :
            # take delta from the other lane
            leftx_current+=delta_right
        elif len(nonzero_left_inds) == 0:
            left_flag=False
        if (len(nonzero_right_inds) < minpix) :
            #take delta from the other lane
            rightx_current += delta_left

        elif len(nonzero_right_inds) == 0:
            right_flag=False

    # make list from list of lists
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    # extract lanes pixels positions
    if len(left_lane_inds)>0 :

        #print('left_lane_inds:', list(map(int,left_lane_inds)))
        left_lane_inds=list(map(int,left_lane_inds))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

    if len(right_lane_inds) > 0:
        right_lane_inds = list(map(int, right_lane_inds))
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    save_img_as(fname, '../output_images/', '_rect', output_img)

    return leftx, lefty, rightx, righty, output_img

#################################################################################################################
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
#################################################################################################################


def fit_polynomial_from_lane_pixels(search_from_previous_line_detected,l_line,r_line,warped_input_img):

    #rect_search is a flag if it is true this means lines are fitted from rectangle boxes search if not, the line are fitted from previous line
    if not search_from_previous_line_detected :
        left_x, left_y, right_x, right_y, out_img = find_lane_pixels(warped_input_img, nwindows=9, margin=100, minpix=50)
    elif search_from_previous_line_detected :
        left_x, left_y, right_x, right_y,out_img=search_around_poly(warped_input_img, l_line, r_line, margin=100)
    #print (left_y)
    #print ("-----------")
    #print(left_x)
    # Fit a second order polynomial for each lane
    left_fit = np.polyfit(left_y, left_x, 2)
    #print('left_fit',left_fit)
    right_fit = np.polyfit(right_y, right_x, 2)
    #print('right_fit',right_fit)

    ysize = warped_input_img.shape[0]
    #print ('ysize',ysize)
    # generate pixels coordinates for plotting
    ploty = np.linspace(0, ysize - 1, ysize)
    try:
        left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]


    except ValueError:
        print("Unable to fit lane!")
        left_fit_x = 1 * ploty ** 2 + 1 * ploty
    l_line.update_line(True, left_fit_x, left_fit, None, None, left_fit_x, ploty)
    try:
        right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    except ValueError:
        print("Unable to fit lane!")
        right_fit_x = 1 * ploty ** 2 + 1 * ploty
    r_line.update_line(True,right_fit_x,right_fit,None,None,right_fit_x,ploty)

    # apply colors to pixels of interest
    #out_img[left_y, left_x] = [255, 0, 0]
    #out_img[right_y, right_x] = [0, 0, 255]

    # plot lane lines
    # plt.plot(left_fit_x,ploty,color='yellow')
    # plt.plot(right_fit_x,ploty,color='yellow')

    # draw left and right lanes
    #create output img
    empty=np.zeros_like(warped_input_img)
    output_img=np.dstack((empty,empty,empty))*255
    verts_left = np.array(list(zip(left_fit_x.astype(int), ploty.astype(int))))
    cv2.polylines(output_img, [verts_left], False, (0, 0, 255), thickness=5)
    verts_right = np.array(list(zip(right_fit_x.astype(int), ploty.astype(int))))
    cv2.polylines(output_img, [verts_right], False, (0, 0, 255), thickness=5)
    colored_input=np.dstack((warped_input_img,warped_input_img,warped_input_img))*255
    #print(out_img.shape, warped_input_img.shape)
    # if search_from_previous_line_detected:
    #
    #     output_img= weighted_img(colored_input, out_img, α=0.8, β=1., γ=0.)
    #     save_img_as(fname, './Output/', '_test', out_img)
    # save image
    #fpath = generate_new_fpath(fname, './demok/', '-searchLine')
    #mpimg.imsave(fpath, output_img)

    return l_line,r_line,output_img

#################################################################################################################

def search_around_poly(img,l_line,r_line,margin):

    left_poly=l_line.current_fit
    right_poly=r_line.current_fit

    output_img=np.dstack((img,img,img))*255
    # locate lanes

    # identify all active pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])



    left_lane_inds = list()
    right_lane_inds = list()

    # define windows boundaries
    # identify non zero (active) pixels within search area (rectangle)
    nonzero_left_inds = list(((nonzerox <=left_poly[0]*(nonzeroy**2)+left_poly[1]*nonzeroy+left_poly[2]+margin ) & (nonzerox >=left_poly[0]*(nonzeroy**2)+left_poly[1]*nonzeroy+left_poly[2]-margin )).nonzero()[0])
    #print('nonzero_left_inds',nonzero_left_inds)
    nonzero_right_inds = list(((nonzerox <=right_poly[0]*(nonzeroy**2)+right_poly[1]*nonzeroy+right_poly[2]+margin ) & (nonzerox >=right_poly[0]*(nonzeroy**2)+right_poly[1]*nonzeroy+right_poly[2]-margin )).nonzero()[0])
    #print('nonzero_right_inds', nonzero_right_inds)
    # append to list for each lane
    #print (type(left_lane_inds),type(nonzero_left_inds))
    left_lane_inds.append(nonzero_left_inds)
    right_lane_inds.append(nonzero_right_inds)


    # extract lanes pixels positions
    left_lane_inds=np.concatenate(left_lane_inds)
    #print('-----------------------------------left_lane_inds', left_lane_inds[5])
    #print('-----------------------------------left_lane_inds dim', left_lane_inds.shape)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    right_lane_inds=np.concatenate(right_lane_inds)
    #print('-----------------------------------right_lane_inds', right_lane_inds[5])
    #print('-----------------------------------right_lane_inds', right_lane_inds.shape)

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    out_img = np.dstack((img, img, img)) * 255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #print(rightx)
    #print (rightx[0])

    return leftx, lefty, rightx, righty,out_img

#################################################################################################################
def fit_polynomial_from_poly_pixels(warped_input_img,left_poly,right_poly,margin):
    left_x, left_y, right_x, right_y=search_around_poly(img,left_poly,right_poly,margin)
    #print (left_y)
    #print ("-----------")
    #print(left_x)
    # Fit a second order polynomial for each lane
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    ysize = warped_input_img.shape[0]
    # generate pixels coordinates for plotting
    ploty = np.linspace(0, ysize - 1, ysize)
    try:
        left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except ValueError:
        print("Unable to fit lane!")
        left_fit_x = 1 * ploty ** 2 + 1 * ploty
        right_fit_x = 1 * ploty ** 2 + 1 * ploty


    # plot lane lines
    # plt.plot(left_fit_x,ploty,color='yellow')
    # plt.plot(right_fit_x,ploty,color='yellow')

    # draw left and right lanes
    #create output img
    empty=np.zeros_like(warped_input_img)
    output_img=np.dstack((empty,empty,empty))*255
    verts_left = np.array(list(zip(left_fit_x.astype(int), ploty.astype(int))))
    cv2.polylines(output_img, [verts_left], False, (0, 255, 0), thickness=5)
    verts_right = np.array(list(zip(right_fit_x.astype(int), ploty.astype(int))))
    cv2.polylines(output_img, [verts_right], False, (255, 0, 0), thickness=5)

    return output_img

#################################################################################################################

def gaussian_blur(image, kernel=5):
    '''
    this routine applies blur to reduce noise in images
    '''
    blurred = cv2.GaussianBlur(image, (kernel,kernel), 0)
    return blurred

#################################################################################################################

def evaluate_curvature_pixels(img,l_line,r_line,position=1):
    #factor is defining at which position the evaluation is done 0 (top of image) 1 (is the bottom)
    conv_factor_y=30/720
    conv_factor_x=3.7/700
    y_eval_rw=int(img.shape[0]*position*conv_factor_y) # at which y we are looking

    ploty=l_line.ally
    #left line curvature calculation
    left_x=l_line.allx
    left_fit_rw=np.polyfit(ploty*conv_factor_y,left_x*conv_factor_x,2)
    left_curverad = int(((1 + (2 * left_fit_rw[0] * y_eval_rw + left_fit_rw[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_rw[0]))

    right_x = r_line.allx
    right_fit_rw=np.polyfit(ploty*conv_factor_y,right_x*conv_factor_x,2)
    right_curverad = int (((1 + (2 * right_fit_rw[0] * y_eval_rw + right_fit_rw[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_rw[0]))


    return left_curverad,right_curverad

#################################################################################################################

def fill_inter_lanes_area(original_undist_img,warped_img,l_line,r_line,transf_mtx_inv):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped_img).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


    left_fitx=l_line.allx
    right_fitx=r_line.allx
    ploty=r_line.ally

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #print(color_warp.shape)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, transf_mtx_inv, (warped_img.shape[1], warped_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_undist_img, 1, newwarp, 0.3, 0)

    return result
#################################################################################################################
#################################################################################################################
#################################################################################################################


#get images from folder
images=glb.glob('./*.jpg')

#create folder to save the new img in
new_dir = './undist/'
create_new_folder (new_dir)

#load saved mtx/dist data  (this is hardoded)
dbpath='../camera_cal/undistorted/mtx_dist'
mtx,dist=load_mtx_dist(dbpath)

#load transform Matrix (using first image)
dbpath='./transform_matrix/mtx_transform'
M,Minv=load_mtx_transform(dbpath)

# Choose a Sobel kernel size and thresholds
ksize=9




for fname in images:
    img=np.copy(mpimg.imread(fname))
    #initialize lines objects
    left_line=Line()
    right_line=Line()

    #undistort image
    undist_img=undist(img,mtx,dist,fname,new_dir)

    img_size = img.shape[1::-1]

    # get warped image input to be used for creating final output only
    warped = cv2.warpPerspective(undist_img, M, img_size)

    ########################################
    # defining thresholds
    thresh_white = (222, 255)
    thresh_yellow = (142, 200)
    thresh_x = (96, 255)
    thresh_y = (0, 255)
    thresh_mag = (10, 255)
    thresh_dir = (0.33, 0.5)

    # Filter noise
    blurr_warped = gaussian_blur(warped, 3)
    #save_img_as(fname, './Output/', '_warped', warped)

    # Apply and combine thresholds
    bin_result_combined=apply_gradient_and_colors_thresholds(blurr_warped, thresh_white, thresh_yellow, ksize, thresh_x, thresh_y,thresh_mag, thresh_dir)
    #save_img_as(fname, './Output/', '_bin_result_combined', bin_result_combined)

    # Fit polynomials to get lane lines
    search_from_previous_line = False
    left_line,right_line,lines=fit_polynomial_from_lane_pixels(search_from_previous_line,left_line,right_line,bin_result_combined)
    lines=None
    # Use actual lines to search for line (this will be done as a test here)
    search_from_previous_line = True
    left_line, right_line, lines = fit_polynomial_from_lane_pixels(search_from_previous_line, left_line,right_line, bin_result_combined)

    # draw lines in blue on top of the warped image
    out_img= weighted_img(blurr_warped, lines, α=0.8, β=1., γ=0.)



    pos=1 #bottom of image
    left_curv,right_curv=evaluate_curvature_pixels(blurr_warped, left_line, right_line, pos)
    print ('left curvature [m]:',left_curv,'right curvature [m]:',right_curv)


    img_with_filling= fill_inter_lanes_area(img,blurr_warped,left_line, right_line,Minv)


    #save image
    save_img_as(fname,'../output_images/', '_output_smoothed', img_with_filling)



