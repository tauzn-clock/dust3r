import os
#Set directory to dust3r
os.chdir("/dust3r")
print(os.getcwd())

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
INIT_FRAMES = 10
RECURRING_FRAMES = 5

device = 'cuda'
batch_size = 15
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
output = inference_with_mask(pairs, model, device, masks, batch_size=batch_size)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()

#Create transform file
#TODO: Per frame camera model?
transform = {}
transform["camera_model"] = "OPENCV"

averge_focal = focals.sum()/len(focals)
transform["fl_x"] = averge_focal.item()
transform["fl_y"] = averge_focal.item()

#Find size of images
img = Image.open(images_array[0])
width, height = img.size
transform["w"] = width
transform["h"] = height
transform["c_x"] = width//2
transform["c_y"] = height//2

transform["frames"] = []

for i in range(len(poses)):
    if not((confidence_mask[i]==0).all()):
        frame = {}
        frame["file_path"] = "/".join(images_array[i].split("/")[-2:])
        frame["transform_matrix"] = poses[i].detach().cpu().numpy().tolist()
        frame["mask_path"] = "/".join(masks_array[i].split("/")[-2:])
        transform["frames"].append(frame)
    else:
        print("No confidence in Frame {}".format(i))

#Save transform file
with open("{}/transforms.json".format(DATA_PATH), 'w') as f:
    json.dump(transform, f, indent=4)