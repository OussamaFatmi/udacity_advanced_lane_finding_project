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

#################################################################################################################
def get_img_obj_points(images,nx, ny, saveas_dir):
    # create global mapping points (for all images)
    objpoints = []  # 3D coord from real chessboard
    imgpoints = []  # 2D coord from images
    # initialize obj pts (this list is to be used the same for every image)
    objpts = np.zeros(((ny * nx), 3), np.float32)
    objpts[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in images:
        # prepare path to save image with corners drwaing

        base = os.path.basename(fname)
        new_fname = os.path.splitext(base)[0] + 'wCorners' + os.path.splitext(base)[1]
        new_fpath = saveas_dir + new_fname

        # read image
        img = mpimg.imread(fname)

        # convert img to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objpts)
            # draw corners and export images
            img_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            mpimg.imsave(new_fpath, img_corners)

    return imgpoints, objpoints

#################################################################################################################
def calibrate_undist(images, imgpoints, objpoints, saveas_dir)
    # calibrate
    img_size = mpimg.imread(images[0]).shape[1::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # save mtx and dist in binary format
    db_file = open(saveas_dir + 'mtx_dist', 'wb')
    db = {}
    db['mtx'] = mtx
    db['dist'] = dist
    pickle.dump(db, db_file)
    db_file.close()

    # undist images
    for fname in images:
        # prepare path to save image with corners drwaing
        base = os.path.basename(fname)
        new_fname = os.path.splitext(base)[0] + 'undist' + os.path.splitext(base)[1]
        new_fpath = saveas_dir + new_fname
        # read image
        img = mpimg.imread(fname)
        # undistort
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        mpimg.imsave(new_fpath, undist)

    return
#################################################################################################################
#################################################################################################################
#################################################################################################################

#initialize chessboard dimension (corners)
nx=9
ny=6
#get images from folder
images=glb.glob('./calibration*.jpg')
#create folder to save the new img in
new_dir = './withCorners/'
create_new_folder (new_dir)
#get image points and object points
imgpoints, objpoints=get_img_obj_points(images, nx, ny, saveas_dir)

###########create undistorted chessboard images and export mtx and dist data###########
#create folder to save the new img in
new_dir = './undistorted/'
create_new_folder (new_dir)

get_img_obj_points(images,nx, ny, new_dir)