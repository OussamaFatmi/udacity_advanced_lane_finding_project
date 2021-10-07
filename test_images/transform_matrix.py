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
def save_transform_matrix(img_filename, saveas_dir):
    img = np.copy(mpimg.imread(img_filename))
    img_size = img.shape[1::-1]
    # prepare source and destination points to calculate the transform matrix
    dim_x = img_size[0]
    pt1 = (219, 720)
    pt2 = (1110, 720)
    pt3 = (675, 442)
    pt4 = (602, 442)
    pts = (pt1, pt2, pt3, pt4)
    src = np.float32(pts).reshape(-1, 2)
    dst = np.copy(src)
    dst[0][0] = 400
    dst[1][0] = dim_x - 400

    dst[3][0] = 400
    dst[2][0] = dim_x - 400
    dst[3][1] = 0
    dst[2][1] = 0

    # calculate transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # calculate inverse transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # save M and Minv in binary format
    db_file = open(saveas_dir + 'mtx_transform', 'wb')
    db = {}
    db['TM'] = M
    db['TMinv'] = Minv

    pickle.dump(db, db_file)
    db_file.close()


    return


#get images from folder
images=glb.glob('./*.jpg')

#create folder to save the new img in
new_dir = './transform_matrix/'
create_new_folder (new_dir)

#caluculate transform Matrix (using first image)
test=images[0]

save_transform_matrix(test, new_dir)

