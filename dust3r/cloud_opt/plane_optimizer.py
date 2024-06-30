# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Slower implementation of the global alignment that allows to freeze partial poses/intrinsics
# --------------------------------------------------------
import torch
import roma

from dust3r.cloud_opt.modular_optimizer import ModularPointCloudOptimizer
from dust3r.utils.geometry import geotrf
from dust3r.cloud_opt.commons import edge_str
from dust3r.cloud_opt.commons import signed_expm1


class PlanePointCloudOptimizer (ModularPointCloudOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Unlike PointCloudOptimizer, you can fix parts of the optimization process (partial poses/intrinsics)
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """
    def __init__(self, *args, 
                 weight_focal = 0.1, 
                 weight_z = 0.1, 
                 weight_rot = 0.001, 
                 weight_trans_smoothness = 0.01, 
                 weight_rot_smoothness = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_z = weight_z
        self.weight_focal = weight_focal
        self.weight_rot = weight_rot
        self.weight_trans_smoothness = weight_trans_smoothness
        self.weight_rot_smoothness = weight_rot_smoothness

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj

        loss /= self.n_edges  # average over all pairs

        all_focal = torch.stack([focal for focal in self.im_focals])
        loss = loss + self.weight_focal * (all_focal.max() - all_focal.min())

        all_poses = torch.stack([pose for pose in self.im_poses])
        loss = loss + self.weight_z * (all_poses[:,6].max() - all_poses[:,6].min())

        Q = all_poses[:,:4]
        T = signed_expm1(all_poses[:,4:7])
        euler = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        euler = roma.euler.rotmat_to_euler("xyz", euler[:,:3,:3])
        loss = loss + self.weight_rot * (euler[:,0].max() - euler[:,0].min())
        loss = loss + self.weight_rot * (euler[:,1].max() - euler[:,1].min())

        for i in range(len(all_poses)-2):
            loss = loss + self.weight_trans_smoothness * (all_poses[i,4:7] - 2*all_poses[i+1,4:7] + all_poses[i+2,4:7]).abs().mean()
            loss = loss + self.weight_rot_smoothness * (all_poses[i,:4] - 2*all_poses[i+1,:4] + all_poses[i+2,:4]).abs().mean()

        if ret_details:
            return loss, details
        return loss 