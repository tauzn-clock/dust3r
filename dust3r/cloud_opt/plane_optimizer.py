# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Slower implementation of the global alignment that allows to freeze partial poses/intrinsics
# --------------------------------------------------------
import torch
import roma
import scipy.sparse as sp


from dust3r.cloud_opt.modular_optimizer import ModularPointCloudOptimizer
from dust3r.utils.geometry import geotrf
from dust3r.cloud_opt.commons import edge_str
from dust3r.cloud_opt.commons import signed_expm1
import dust3r.cloud_opt.init_im_poses as init_fun
from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores
from dust3r.cloud_opt.base_opt import global_alignment_loop
from dust3r.utils.geometry import geotrf

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

        #all_focal = torch.stack(list(self.im_focals))
        all_focal = self.get_focals().reshape(-1)
        all_poses = torch.stack(list(self.im_poses))
        Q = all_poses[:,:4]
        Q = torch.nn.functional.normalize(Q, p=2, dim=1)
        T = signed_expm1(all_poses[:,4:7])
        tf = roma.RigidUnitQuat(Q, T).normalize()#.to_homogeneous()#.inverse()
        tf_inv = tf.inverse()
        #tf = torch.matmul(tf, self.OPENGL)

        off_z_axis = torch.zeros(len(self.edges)).to(Q.device)

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
            off_z_axis[e] = cur_axis.unsqueeze(0)

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


        loss = loss + self.weight_trans_smoothness * (T[:len(all_poses)-2] - 2*T[1:len(all_poses)-1] + T[2:len(all_poses)]).abs().mean()
        loss = loss + self.weight_rot_smoothness * (Q[:len(all_poses)-2] - 2*Q[1:len(all_poses)-1] + Q[2:len(all_poses)]).abs().mean()
        #for i in range(len(all_poses)-2):
        #    loss = loss + self.weight_trans_smoothness * (T[i] - 2*T[i+1] + T[i+2]).abs().mean() / len(all_poses)
        #    loss = loss + self.weight_rot_smoothness * (Q[i] - 2*Q[i+1] + Q[i+2]).abs().mean() / len(all_poses)

        if ret_details:
            return loss, details
        return loss 
    
    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == 'seq':
            init_sequential_frames(self, niter_PnP=niter_PnP)
        elif init == 'msp' or init == 'mst':
            init_fun.init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
        elif init == 'known_poses':
            init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
                                           niter_PnP=niter_PnP)
        else:
            raise ValueError(f'bad value for {init=}')

        return global_alignment_loop(self, **kw)

@torch.no_grad()
def init_sequential_frames(self, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    pts3d, _, im_focals, im_poses = sequntial_frames(self.imshapes, self.edges,
                                                          self.pred_i, self.pred_j, self.conf_i, self.conf_j, self.im_conf, self.min_conf_thr,
                                                          device, has_im_poses=self.has_im_poses, verbose=self.verbose,
                                                          **kw)
    return init_fun.init_from_pts3d(self, pts3d, im_focals, im_poses)

def sequntial_frames(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,
                    device, has_im_poses=True, niter_PnP=10, verbose=True):
    n_imgs = len(imshapes)

    pts3d = [None] * n_imgs
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs
    msp_edges = [(0, 1)]

    i_j = edge_str(0, 1)
    pts3d[0] = pred_i[i_j].clone()
    pts3d[1] = pred_j[i_j].clone()
    im_poses[0] = torch.eye(4, device=device)
    im_focals[0] = init_fun.estimate_focal(pred_i[i_j])
    
    for i in range(1,n_imgs-1):
        j = i+1
        i_j = edge_str(i, j)
        s, R, T = init_fun.rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])
        trf = init_fun.sRT_to_4x4(s, R, T, device)
        pts3d[j] = geotrf(trf, pred_j[i_j])
        im_poses[j] = init_fun.sRT_to_4x4(1, R, T, device)
        msp_edges.append((i, j))
        if has_im_poses and im_poses[i] is None:
            im_poses[i] = init_fun.sRT_to_4x4(1, R, T, device)   
    print(im_poses[1])

    for i in range(n_imgs-1):
        j = i+1
        if im_focals[i] is None:
            im_focals[i] = init_fun.estimate_focal(pred_i[edge_str(i, j)])

    for i in range(n_imgs):
        if im_poses[i] is None:
            msk = im_conf[i] > min_conf_thr
            res = init_fun.fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
            if res:
                im_focals[i], im_poses[i] = res
        if im_poses[i] is None:
            im_poses[i] = torch.eye(4, device=device)
    im_poses = torch.stack(im_poses)

    print(im_poses[1])
    
    return pts3d, msp_edges, im_focals, im_poses