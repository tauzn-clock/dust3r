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

DATA_PATH = "/dust3r/masked_dust3r/data/jackal_training_data_0"
IMG_FILE_EXTENSION = ".png"
MASK_FILE_EXTENSION = ".png"
GAUSSIAN_SIGMA = 3.0
INIT_FRAMES = 15
RECURRING_FRAMES = 5
TOTAL_IMGS = 50

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
# you can put the path to a local checkpoint in model_name if needed
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)


# load_images can take a list of images or a directory
images_array = []
masks_array = []

for i in range(INIT_FRAMES):
    images_array.append(os.path.join(DATA_PATH,"masked_images/{}{}".format(i,IMG_FILE_EXTENSION)))
    masks_array.append(os.path.join(DATA_PATH,"masks/{}{}".format(i,MASK_FILE_EXTENSION)))
images = load_images(images_array, size=512, verbose=True)

masks = []

for i in range(len(masks_array)):
    mask = Image.open(masks_array[i]).convert('L')
    _,_,H,W = images[i]["img"].shape
    mask = mask.resize((W,H))

    mask = np.array(mask)
    mask = torch.tensor(mask).to(device)/255
    masks.append(mask)

pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference_with_mask(pairs, model, device, masks, GAUSSIAN_SIGMA, batch_size=batch_size)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
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
    else:
        print("No confidence in Frame {}".format(i))

#Save transform file
with open("{}/transforms.json".format(DATA_PATH), 'w') as f:
    json.dump(transforms, f, indent=4)

for new_img_index in range(INIT_FRAMES, TOTAL_IMGS):
    print("Looking at frame {}...".format(new_img_index))
    images_array = []
    masks_array = []

    preset_focal = [transforms["fl_x"] for _ in range(RECURRING_FRAMES+1)]
    preset_pose = []
    preset_mask = [True for _ in range(RECURRING_FRAMES+1)]
    preset_mask[-1] = False

    for i in range(-RECURRING_FRAMES,0):
        images_array.append(os.path.join(DATA_PATH,transforms["frames"][i]["file_path"]))
        masks_array.append(os.path.join(DATA_PATH,transforms["frames"][i]["mask_path"]))
        preset_pose.append(np.array(transforms["frames"][i]["transform_matrix"]))
        print("Using {}...".format(transforms["frames"][i]["file_path"]))

    images_array.append(os.path.join(DATA_PATH,"masked_images/{}{}".format(new_img_index,IMG_FILE_EXTENSION)))
    masks_array.append(os.path.join(DATA_PATH,"masks/{}{}".format(new_img_index,MASK_FILE_EXTENSION)))
    preset_pose.append(np.eye(4))

    images = load_images(images_array, size=512, verbose=True)

    masks = []

    for i in range(len(masks_array)):
        mask = Image.open(masks_array[i]).convert('L')
        _,_,H,W = images[i]["img"].shape
        mask = mask.resize((W,H))

        mask = np.array(mask)
        mask = torch.tensor(mask).to(device)/255
        masks.append(mask)
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference_with_mask(pairs, model, device, masks, GAUSSIAN_SIGMA, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer)
    scene.preset_focal(preset_focal, preset_mask)
    scene.preset_pose(preset_pose, preset_mask)

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    if (confidence_masks[-1]!=0).all():
        print("No confidence in Frame {}".format(new_img_index))       

    new_frame = {
        "file_path" : "/".join(images_array[-1].split("/")[-2:]),
        "transform_matrix" : poses[-1].detach().cpu().numpy().tolist(),
        "mask_path" : "/".join(masks_array[-1].split("/")[-2:])
    }
    transforms["frames"].append(new_frame)

    with open("{DATA_PATH}/transforms.json".format(DATA_PATH=DATA_PATH), "w") as f:
        json.dump(transforms, f, indent=4)