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
                 weight_focal = 0, 
                 weight_z = 0, 
                 weight_rot = 0, 
                 weight_trans_smoothness = 0, 
                 weight_rot_smoothness = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_z = weight_z
        self.weight_focal = weight_focal
        self.weight_rot = weight_rot
        self.weight_trans_smoothness = weight_trans_smoothness
        self.weight_rot_smoothness = weight_rot_smoothness
        self.OPENGL = torch.tensor([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]], dtype=torch.float32).to("cuda") # TODO: Redefine location 

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        all_focal = torch.stack(list(self.im_focals))
        all_poses = torch.stack(list(self.im_poses))
        Q = all_poses[:,:4]
        Q = torch.nn.functional.normalize(Q, p=2, dim=1)
        T = signed_expm1(all_poses[:,4:7])
        tf = roma.RigidUnitQuat(Q, T).normalize()#.to_homogeneous()#.inverse()
        tf_inv = tf.inverse()
        #tf = torch.matmul(tf, self.OPENGL)

        off_z_axis = torch.tensor([]).to(Q.device)

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

            cur_axis = 1 - torch.nn.functional.normalize((tf[j] @ tf_inv[i]).linear[:3],p=2, dim=0, eps=1e-12)[2].abs()
            off_z_axis = torch.cat((off_z_axis, cur_axis.unsqueeze(0)))

            if ret_details:
                details[i, j] = li + lj

        loss /= self.n_edges  # average over all pairs

        #loss = loss + self.weight_focal * (all_focal.max() - all_focal.min()) ** 2
        loss = loss + self.weight_focal * all_focal.var()

        #loss = loss + self.weight_z * (T[:,2].max() - T[:,2].min()) ** 2
        loss = loss + self.weight_z * T[:,2].var()

        loss = loss + self.weight_rot * off_z_axis.mean()

        #x_w = tf.linear[:,0]
        #y_w = tf.linear[:,1]

        #loss = loss + self.weight_rot * (x_w.max() - x_w.min())
        #loss = loss + self.weight_rot * (y_w.max() - y_w.min())
        #euler = roma.euler.rotmat_to_euler("xyz", tf[:,:3,:3])
        #loss = loss + self.weight_rot * (euler[:,0].max() - euler[:,0].min())
        #loss = loss + self.weight_rot * (euler[:,1].max() - euler[:,1].min())
        #loss = loss + self.weight_rot * euler[:,0].var()
        #loss = loss + self.weight_rot * euler[:,1].var()

        for i in range(len(all_poses)-2):
            loss = loss + self.weight_trans_smoothness * (T[i] - 2*T[i+1] + T[i+2]).abs().mean()
        #    loss = loss + self.weight_rot_smoothness * (Q[i] - 2*Q[i+1] + Q[i+2]).abs().mean()

        if ret_details:
            return loss, details
        return loss 