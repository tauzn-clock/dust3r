import os
os.chdir("/dust3r")
print(os.getcwd())

import sys
sys.path.append('/dust3r')

import open3d as o3d
import numpy as np

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

device = 'cuda'
batch_size = 1
schedule = 'cosine'
lr = 0.01
niter = 300

DATA_PATH = "/scratch/indoor_short"

model_name = "/scratchdata/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

images_array = []

for i in range(2):
    images_array.append(os.path.join(DATA_PATH,"rgb/{}.png".format(i)))
images = load_images(images_array, size=512, verbose=True)

pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=batch_size)

scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

imgs = scene.imgs
focals = scene.get_focals()
poses = scene.get_im_poses()
pts3d = scene.get_pts3d()
confidence_masks = scene.get_masks()

pcd = o3d.geometry.PointCloud()

for i in range(len(pts3d)):
    pointcloud = pts3d[i].detach().cpu().numpy()
    pointcloud = pointcloud.reshape(-1, 3)
    color = imgs[i].reshape(-1, 3)
    
    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(pointcloud)
    tmp_pcd.colors = o3d.utility.Vector3dVector(color)
    tmp_pcd.voxel_down_sample(voxel_size=0.01)
    
    pcd += tmp_pcd
    
o3d.io.write_point_cloud("/scratchdata/pointcloud.ply", pcd)