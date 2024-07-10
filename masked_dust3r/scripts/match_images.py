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
import torch
import json
import open3d as o3d

from dust3r.inference import inference_with_mask, create_gaussian_kernel
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.cloud_opt.base_opt import global_alignment_loop
from masked_dust3r.scripts.utils.math import *
from masked_dust3r.scripts.utils.image import *


DATA_PATH = "/dust3r/masked_dust3r/data/jackal_color_wheel_training_data_1"
IMG_FILE_EXTENSION = ".png"
MASK_FILE_EXTENSION = ".png"

INIT_FRAMES = 20
NEW_FRAMES = 5
PREVIOUS_FRAMES = 10
TOTAL_FRAMES = 300

INIT_WEIGHT_FOCAL = 0.1
INIT_WEIGHT_Z = 0.1 
INIT_WEIGHT_ROT = 0.01
INIT_WEIGHT_TRANS_SMOOTHNESS = 0.0001 
INIT_WEIGHT_ROT_SMOOTHNESS = 0.001 * 0

NEW_WEIGHT_FOCAL = 0.1
NEW_WEIGHT_Z = 0.1
NEW_WEIGHT_ROT = 0.01
NEW_WEIGHT_TRANS_SMOOTHNESS = 0.0001
NEW_WEIGHT_ROT_SMOOTHNESS = 0.00001 * 0

USE_COMMON_INTRINSICS = True
IS_FOCAL_FIXED = True

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300 

GAUSSIAN_SIGMA = 51.0
SIZE = int(GAUSSIAN_SIGMA * 3)

kernel = create_gaussian_kernel(SIZE, GAUSSIAN_SIGMA).to(device)

SIZE = 11
kernel = torch.ones(SIZE, SIZE).to(device)

# Load the model

model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# you can put the path to a local checkpoint in model_name if needed
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

#STEP 1: Perform initial match using INIT_FRAMES frames
images_array = []
masks_array = []

for i in range(INIT_FRAMES):
    images_array.append(os.path.join(DATA_PATH,"masked_images/{}{}".format(i,IMG_FILE_EXTENSION)))
    masks_array.append(os.path.join(DATA_PATH,"masks/{}{}".format(i,MASK_FILE_EXTENSION)))
images = load_images(images_array, size=512, verbose=True)
_,_,H,W = images[0]["img"].shape
masks = load_masks(masks_array, H, W, device)

pairs = make_pairs(images, scene_graph='swin-5', prefilter=None, symmetrize=True)
output = inference_with_mask(pairs, model, device, masks, kernel, batch_size=batch_size)
del pairs

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PlanePointCloudOptimizer, 
                        weight_focal = INIT_WEIGHT_FOCAL,
                        weight_z = INIT_WEIGHT_Z ,
                        weight_rot = INIT_WEIGHT_ROT  ,
                        weight_trans_smoothness = INIT_WEIGHT_TRANS_SMOOTHNESS,
                        weight_rot_smoothness = INIT_WEIGHT_ROT_SMOOTHNESS)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()
intrinsics = scene.get_intrinsics()

#Create transform file

img = Image.open(images_array[0])
width, height = img.size
RESCALE_FACTOR  = width/512

transforms = {}
transforms["camera_model"] = "OPENCV"
if USE_COMMON_INTRINSICS:
    intrinsic_mean = intrinsics.mean(dim=0)
    transforms["fl_x"] = intrinsic_mean[0,0].item() * RESCALE_FACTOR
    transforms["fl_y"] = intrinsic_mean[1,1].item() * RESCALE_FACTOR
    transforms["w"] = width 
    transforms["h"] = height 
    transforms["cx"] = intrinsic_mean[0,2].item() * RESCALE_FACTOR
    transforms["cy"] = intrinsic_mean[1,2].item() * RESCALE_FACTOR

transforms["frames"] = []

OPENGL = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

for i in range(len(poses)):
    if not((confidence_masks[i]==0).all()):
        frame = {}
        frame["file_path"] = "/".join(images_array[i].split("/")[-2:])
        #frame["transform_matrix"] = np.linalg.inv(poses[i].detach().cpu().numpy()).tolist()
        frame["transform_matrix"] = np.dot(poses[i].detach().cpu().numpy(),OPENGL).tolist()
        frame["mask_path"] = "/".join(masks_array[i].split("/")[-2:])
        transforms["frames"].append(frame)
        
        if not USE_COMMON_INTRINSICS:
            frame["fl_x"] = intrinsics[i,0,0].item() * RESCALE_FACTOR
            frame["fl_y"] = intrinsics[i,1,1].item() * RESCALE_FACTOR
            frame["cx"] = intrinsics[i,0,2].item() * RESCALE_FACTOR
            frame["cy"] = intrinsics[i,1,2].item() * RESCALE_FACTOR
            img = Image.open(images_array[i])
            width, height = img.size
            transforms["w"] = width 
            transforms["h"] = height 

#Save transform file
with open("{}/transforms.json".format(DATA_PATH), 'w') as f:
    json.dump(transforms, f, indent=4)

# STEP 2: Match Future Frames

for start_frame_index in range(INIT_FRAMES, TOTAL_FRAMES, NEW_FRAMES):
    images_array = []
    masks_array = []

    preset_focal = [transforms["fl_x"]/RESCALE_FACTOR for _ in range(PREVIOUS_FRAMES+NEW_FRAMES)]
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
    output = inference_with_mask(pairs, model, device, masks, kernel, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PlanePointCloudOptimizer, 
                            weight_focal = NEW_WEIGHT_FOCAL,
                            weight_z = NEW_WEIGHT_Z ,
                            weight_rot = NEW_WEIGHT_ROT ,
                            weight_trans_smoothness = NEW_WEIGHT_TRANS_SMOOTHNESS,
                            weight_rot_smoothness = NEW_WEIGHT_ROT_SMOOTHNESS)
    scene.preset_focal(preset_focal, [True for _ in range(PREVIOUS_FRAMES+NEW_FRAMES)])
    scene.preset_pose(preset_pose, preset_mask)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    intrinsics = scene.get_intrinsics()

    for i in range(PREVIOUS_FRAMES, PREVIOUS_FRAMES+NEW_FRAMES):
        if not((confidence_masks[i]==0).all()):
            frame = {}
            frame["file_path"] = "/".join(images_array[i].split("/")[-2:])
            #frame["transform_matrix"] = np.linalg.inv(poses[i].detach().cpu().numpy()).tolist()
            frame["transform_matrix"] = np.dot(poses[i].detach().cpu().numpy(),OPENGL).tolist()
            frame["mask_path"] = "/".join(masks_array[i].split("/")[-2:])
            transforms["frames"].append(frame)
            
            if not USE_COMMON_INTRINSICS:
                frame["fl_x"] = intrinsics[i,0,0].item() * RESCALE_FACTOR
                frame["fl_y"] = intrinsics[i,1,1].item() * RESCALE_FACTOR
                frame["cx"] = intrinsics[i,0,2].item() * RESCALE_FACTOR
                frame["cy"] = intrinsics[i,1,2].item() * RESCALE_FACTOR
                img = Image.open(images_array[i])
                width, height = img.size
                transforms["w"] = width 
                transforms["h"] = height 

    with open(f"{DATA_PATH}/transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)