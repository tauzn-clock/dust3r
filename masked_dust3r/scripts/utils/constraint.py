import numpy as np
from .math import *

def rotate_best_fit_plane(frames):
    camera_origin = []
    for i in range(len(frames)):
        tf = frames[i]["transform_matrix"]
        tf = np.array(tf).reshape(4, 4)
        camera_origin.append(tf[:3, 3])
    camera_origin = np.array(camera_origin)

    a, b, c, d = best_fit_plane(camera_origin)
    R_TF = fit_to_xy_plane(a, b, c, d)

    for i in range(len(frames)):
        tf = frames[i]["transform_matrix"]
        tf = np.array(tf).reshape(4, 4)
        tf = R_TF @ tf
        frames[i]["transform_matrix"] = tf.tolist()

    return frames

def zero_z(frames):
    for i in range(len(frames)):
        tf = frames[i]["transform_matrix"]
        tf = np.array(tf).reshape(4, 4)
        tf[2, 3] = 0
        frames[i]["transform_matrix"] = tf.tolist()
    
    return frames

def fix_roll_yaw(frames):
    #Assumes plane of best fit is parallel to xy plane 
    pass