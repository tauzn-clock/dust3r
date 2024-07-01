#!/bin/python3 python3.10

import os
#Set directory to dust3r
os.chdir("/dust3r")
print(os.getcwd())

import sys
sys.path.append('/dust3r')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import open3d as o3d
import torch
import json

from dust3r.inference import inference_with_mask
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from masked_dust3r.scripts.utils.math import *
from masked_dust3r.scripts.utils.image import *


DATA_PATH = "/dust3r/masked_dust3r/data/jackal_training_data_0"
IMG_FILE_EXTENSION = ".png"
MASK_FILE_EXTENSION = ".png"
GAUSSIAN_SIGMA = 1.0
INIT_FRAMES = 50
NEW_FRAMES = 10
PREVIOUS_FRAMES = 10
TOTAL_FRAMES = 300

IS_FOCAL_FIXED = True
FOCAL_LENGTH = 4.74

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

# Load the model

model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# you can put the path to a local checkpoint in model_name if needed
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
#STEP 1: Perform initial match using INIT_FRAMES frames
 
images_array = []
masks_array = []

for i in range(0,50):
    images_array.append(os.path.join(DATA_PATH,"masked_images/{}{}".format(i,IMG_FILE_EXTENSION)))
    masks_array.append(os.path.join(DATA_PATH,"masks/{}{}".format(i,MASK_FILE_EXTENSION)))
images = load_images(images_array, size=512, verbose=True)

_,_,H,W = images[i]["img"].shape

masks = load_masks(masks_array, H, W, device)

pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)
output = inference_with_mask(pairs, model, device, masks, GAUSSIAN_SIGMA, batch_size=batch_size)

init_scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
loss = init_scene.compute_global_alignment(init="mst", niter=niter, schedule='cosine', lr=lr)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PlanePointCloudOptimizer, 
                       weight_focal = 1, 
                       weight_z = 0.1, 
                       weight_rot = 0.1, 
                       weight_trans_smoothness = 0.001,
                       weight_rot_smoothness = 0.001)
scene.im_poses = calculate_new_params(init_scene.im_poses,device)
scene.im_focals = init_scene.im_focals
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()

#Create transform file
#TODO: Per frame camera model?
transforms = {}
transforms["camera_model"] = "OPENCV"

averge_focal = focals.sum()/len(focals)
transforms["fl_x"] = averge_focal.item()
transforms["fl_y"] = averge_focal.item()

#Find size of images
img = Image.open(images_array[0])
width, height = img.size
transforms["w"] = width
transforms["h"] = height
transforms["cx"] = width//2
transforms["cy"] = height//2

transforms["frames"] = []

for i in range(len(poses)):
    if not((confidence_masks[i]==0).all()):
        frame = {}
        frame["file_path"] = "/".join(images_array[i].split("/")[-2:])
        frame["transform_matrix"] = poses[i].detach().cpu().numpy().tolist()
        frame["mask_path"] = "/".join(masks_array[i].split("/")[-2:])
        transforms["frames"].append(frame)

#Save transform file
with open("{}/transforms.json".format(DATA_PATH), 'w') as f:
    json.dump(transforms, f, indent=4)

# STEP 2: Match Future Frames

for start_frame_index in range(INIT_FRAMES, TOTAL_FRAMES, NEW_FRAMES):
    images_array = []
    masks_array = []

    preset_focal = [transforms["fl_x"] for _ in range(PREVIOUS_FRAMES+NEW_FRAMES)]
    preset_pose = []
    preset_mask = [True for _ in range(PREVIOUS_FRAMES+NEW_FRAMES)]
    preset_mask[PREVIOUS_FRAMES:] = [False for _ in range(NEW_FRAMES)]

    for i in range(len(transforms["frames"])-PREVIOUS_FRAMES, len(transforms["frames"])):
        images_array.append(os.path.join(DATA_PATH,transforms["frames"][i]["file_path"]))
        masks_array.append(os.path.join(DATA_PATH,transforms["frames"][i]["mask_path"]))
        preset_pose.append(np.array(transforms["frames"][i]["transform_matrix"]))
        print("Refering to {}...".format(transforms["frames"][i]["file_path"]))

    last_known_pose = preset_pose[-1]

    for i in range(start_frame_index, start_frame_index + NEW_FRAMES):
        images_array.append(os.path.join(DATA_PATH,"masked_images/{}{}".format(i,IMG_FILE_EXTENSION)))
        masks_array.append(os.path.join(DATA_PATH,"masks/{}{}".format(i,MASK_FILE_EXTENSION)))
        preset_pose.append(last_known_pose)
        print("Estimating for {}...".format(os.path.join(DATA_PATH,"masked_images/{}{}".format(i,IMG_FILE_EXTENSION))))

    images = load_images(images_array, size=512, verbose=True)
    _,_,H,W = images[0]["img"].shape
    masks = load_masks(masks_array, H, W, device)
    
    pairs = make_pairs(images, scene_graph='swin-{}'.format(PREVIOUS_FRAMES), prefilter=None, symmetrize=True)
    output = inference_with_mask(pairs, model, device, masks, GAUSSIAN_SIGMA, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PlanePointCloudOptimizer, 
                       weight_focal = 1, 
                       weight_z = 0.1, 
                       weight_rot = 0.1, 
                       weight_trans_smoothness = 0.001,
                       weight_rot_smoothness = 0.001)
    scene.preset_focal(preset_focal, [True for _ in range(PREVIOUS_FRAMES+NEW_FRAMES)])
    scene.preset_pose(preset_pose, preset_mask)

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    for i in range(PREVIOUS_FRAMES, PREVIOUS_FRAMES+NEW_FRAMES):
        new_frame = {
            "file_path" : "/".join(images_array[i].split("/")[-2:]),
            "transform_matrix" : poses[i].tolist(),
            "mask_path" : "/".join(masks_array[i].split("/")[-2:])
        }
        if confidence_masks[i].sum() > 0:
            transforms["frames"].append(new_frame)
        else:
            print("Reject frame {} due to low confidence".format(i))

    with open(f"{DATA_PATH}/transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)