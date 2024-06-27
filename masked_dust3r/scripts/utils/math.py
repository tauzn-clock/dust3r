import numpy as np

def best_fit_plane(points):
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

