import torch
import numpy as np
import roma

from dust3r.cloud_opt.commons import signed_expm1


def best_fit_plane(points):
    if type(points) == torch.Tensor:        
        # Calculate the centroid of the point cloud
        centroid = torch.mean(points, dim=0)
        
        # Center the points by subtracting the centroid
        centered_points = points - centroid
        
        # Perform SVD on the centered points
        _, _, vh = torch.linalg.svd(centered_points)
        
        # The normal vector to the plane is the last row of vh (smallest singular value)
        normal_vector = vh[-1, :]
        
        # Plane equation coefficients
        a, b, c = normal_vector
        d = -torch.dot(normal_vector, centroid)
        
        return a.item(), b.item(), c.item(), d.item()

    # Convert the point list to a numpy array
    points = np.array(points)
    
    # Calculate the centroid of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Center the points by subtracting the centroid
    centered_points = points - centroid
    
    # Perform SVD on the centered points
    _, _, vh = np.linalg.svd(centered_points)

    # The normal vector to the plane is the last row of vh (smallest singular value)
    normal_vector = vh[-1, :]
    
    # Plane equation coefficients
    a, b, c = normal_vector
    d = -np.dot(normal_vector, centroid)
    
    return a, b, c, d

def fit_to_xy_plane(a,b,c,d):
    def calc_cos_phi(a, b, c):
        return c / (a*a + b*b + c*c)**0.5


    def calc_sin_phi(a, b, c):
        return ((a*a + b*b) / (a*a + b*b + c*c)) ** 0.5


    def calc_u1(a, b, c):
        return b / (a*a + b*b)**0.5


    def calc_u2(a, b, c):
        return -a / (a*a + b*b)**0.5
    
    cos_phi = calc_cos_phi(a, b, c)
    sin_phi = calc_sin_phi(a, b, c)
    u1 = calc_u1(a, b, c)
    u2 = calc_u2(a, b, c)
    R = np.array([
        [cos_phi + u1 * u1 * (1 - cos_phi)  , u1 * u2 * (1 - cos_phi)           , u2 * sin_phi  ,  0            ],
        [u1 * u2 * (1 - cos_phi)            , cos_phi + u2 * u2 * (1 - cos_phi) , -u1 * sin_phi ,  0            ],
        [-u2 * sin_phi                      , u1 * sin_phi                      ,      cos_phi  ,  d / (a**2 + b**2 + c**2)**0.5],
        [0                                  , 0                                 , 0             ,  1            ]
    ])

    return R

def calculate_new_params(current_parameters, device):
    centre = torch.stack([p for p in current_parameters])[:,4:]
    centre = signed_expm1(centre)

    a,b,c,d = best_fit_plane(centre)
    R = fit_to_xy_plane(a,b,c,d)
    R = torch.tensor(R).to(device).float()

    all_poses = torch.stack([p for p in current_parameters])
    Q = all_poses[:,:4]
    T = signed_expm1(all_poses[:,4:7])
    euler = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
    euler = R @ euler

    quat = roma.rotmat_to_unitquat(euler[:,:3,:3])

    new_parameters = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(7, dtype=torch.float32).to(device))] * len(current_parameters))
    for i in range(len(current_parameters)):
        new_parameters[i] = torch.cat([quat[i], inv_signed_expm1(euler[i,:3,3])])

    return new_parameters

def interpolate_pose(p0, p1, step_cnt, device):

    step = (torch.arange(step_cnt+1).to(device) / step_cnt).double()

    #Interpolate translation
    p0_t = p0[:3, 3]
    p0_t = torch.tensor(p0_t).to(device).reshape(3, 1)
    p1_t = p1[:3, 3]
    p1_t = torch.tensor(p1_t).to(device).reshape(3, 1)

    t = step * p1_t + (1 - step) * p0_t
    t = t.transpose(0, 1).reshape(-1, 3, 1)

    #Interpolate rotation
    p0_R = p0[:3, :3]
    p0_R = torch.tensor(p0_R).to(device).reshape(3, 3)
    p1_R = p1[:3, :3]
    p1_R = torch.tensor(p1_R).to(device).reshape(3, 3)

    R = roma.utils.rotmat_slerp(p0_R, p1_R, step)

    #Combine into 4x4 matrix
    tf = torch.cat([R, t], dim=2)

    bottom_row = torch.tensor([0, 0, 0, 1]).to(device)
    bottom_row = bottom_row.unsqueeze(0).unsqueeze(0).expand(step_cnt+1, 1, 4)

    tf = torch.cat((tf, bottom_row), dim=1).to(device)

    return tf

def inv_signed_expm1(x):
    sign = torch.sign(x)
    return sign * (torch.log(torch.abs(x)) + 1)