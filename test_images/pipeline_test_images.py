import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob as glb
import os
import cv2
import pickle


#################################################################################################################
def create_new_folder (new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return
def generate_new_fpath (fname,saveas_dir,key_word):
    base = os.path.basename(fname)
    new_fname = os.path.splitext(base)[0] + key_word + os.path.splitext(base)[1]
    new_fpath = saveas_dir + new_fname

    return new_fpath

#################################################################################################################
def load_mtx_dist(fpath):
    # load mtx and dist db

    db_file = open(fpath, 'rb')
    pickle_dist = pickle.load(db_file)
    mtx = pickle_dist['mtx']
    dist = pickle_dist['dist']

    return mtx, dist

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
    grad_binary[(scaled_sobel<thresh[1])&(scaled_sobel>thresh[0])]=1

    return grad_binary
#################################################################################################################

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobel_x=cv2.Sobel(image,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobel_y=cv2.Sobel(image,cv2.CV_64F,0,1,ksize=sobel_kernel)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    scaled_mag=np.uint8(255*grad_mag/(np.max(grad_mag)))

    mag_binary=np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag<mag_thresh[1])&(scaled_mag>mag_thresh[0])]=1
    # Apply threshold
    return mag_binary
#################################################################################################################

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    abs_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel=np.arctan2(np.absolute(sobel_y),np.absolute(sobel_x))

    dir_binary=np.zeros_like(dir_sobel)
    # Apply threshold
    dir_binary[(dir_sobel>thresh[0])&(dir_sobel<thresh[1])]=1
    return dir_binary
#################################################################################################################


#################################################################################################################
#################################################################################################################
#################################################################################################################


#get images from folder
images=glb.glob('./*.jpg')

#create folder to save the new img in
new_dir = './undist/'
create_new_folder (new_dir)

#saved data file path this is hardoded
dbpath='../camera_cal/undistorted/mtx_dist'
mtx,dist=load_mtx_dist(dbpath)

# Choose a Sobel kernel size and thresholds
ksize=3
threshx_max=100
threshx_min=30
sthresh_max=255
sthresh_min=100
thresh_mag_min=30
thresh_mag_max=100
thresh_dir_min=0.7
thresh_dir_max=1.3

for fname in images:
    img=np.copy(mpimg.imread(fname))
    undist_img=undist(img,mtx,dist,fname,new_dir)

    #turn to gray
    gray=cv2.cvtColor(undist_img,cv2.COLOR_RGB2GRAY)
    #turn to hls
    hls=cv2.cvtColor(undist_img,cv2.COLOR_RGB2HLS)
    s_channel=hls[:,:,2]
    l_channel=hls[:,:,1]


    #sobel grad x
    thresh_x=(threshx_min , threshx_max)
    sobel_x=abs_sobel_thresh(s_channel,'x', ksize, thresh_x)
    fpath=generate_new_fpath(fname, './demok/', '_grad_x')
    mpimg.imsave(fpath, sobel_x, cmap='gray')

    # sobel grad y
    sobel_y = abs_sobel_thresh(s_channel, 'y', ksize, thresh_x)
    fpath=generate_new_fpath(fname, './demok/', '_grad_y')
    mpimg.imsave(fpath, sobel_y, cmap='gray')

    #sobel magnitude
    thresh_mag=(thresh_mag_min,thresh_mag_max)
    mag_binary = mag_thresh(s_channel, ksize, thresh_mag)
    fpath=generate_new_fpath(fname, './demok/', '_mag')
    mpimg.imsave(fpath, mag_binary, cmap='gray')
    #sobel_dir
    thresh_dir=(thresh_dir_min,thresh_dir_max)
    dir_binary = dir_threshold(s_channel, ksize, thresh_dir)
    fpath=generate_new_fpath(fname, './demok/', '_dir')
    mpimg.imsave(fpath, dir_binary, cmap='gray')

    fpath = generate_new_fpath(fname, './demo/', '_dir')
    binary_out=np.zeros_like(dir_binary)
    binary_out[(sobel_x==1)&(sobel_y==1)&(mag_binary==1)&(dir_binary==1)]=1


    # #threshold x gradient
    # sx_binary=np.zeros_like(s_channel)
    # sx_binary[(scaled_sobelx<threshx_max)&(scaled_sobelx>threshx_min)]=1
    # #threshold s channel
    s_binary=np.zeros_like(gray)
    s_binary[(s_channel<sthresh_max) & (s_channel>sthresh_min)]=1
    mpimg.imsave(fpath, s_binary, cmap='gray')
    # s_binary[(((s_channel < sthresh_max) & (s_channel > sthresh_min))|
    #          ((scaled_sobelx<threshx_max)&(scaled_sobelx>threshx_min)))&((grad_dir<1.3)&(grad_dir>0.7))]=1

    #color_binary=np.dstack((np.zeros_like(s_channel),sx_binary,s_binary))*255
    #mpimg.imsave(saveas, s_binary,cmap='gray')
