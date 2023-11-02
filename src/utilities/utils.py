from scipy.linalg import expm, logm
from scipy.optimize import least_squares

import numpy as np


def twist2HomogMatrix(twist):
    """
    twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    Input: -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
    Output: -H(4,4): Euclidean transformation matrix (rigid body motion)
    """
    v = twist[:3]  # linear part
    w = twist[3:]   # angular part

    se_matrix = np.concatenate([cross2Matrix(w), v[:, None]], axis=1)
    se_matrix = np.concatenate([se_matrix, np.zeros([1, 4])], axis=0)

    H = expm(se_matrix)

    return H


def HomogMatrix2twist(H):
    """
    HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    Input:
     -H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
     -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]

    Observe that the same H might be represented by different twist vectors
    Here, twist(4:6) is a rotation vector with norm in [0,pi]
    """

    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not
    # skew-symmetric (bad).

    v = se_matrix[:3, 3]

    w = Matrix2Cross(se_matrix[:3, :3])
    twist = np.concatenate([v, w])

    return twist

def cross(a:np.ndarray, b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M

def Matrix2Cross(M):
    """
    Computes the 3D vector x corresponding to an antisymmetric matrix M such that M*y = cross(x,y)
    for all 3D vectors y.
    Input:
     - M(3,3) : antisymmetric matrix
    Output:
     - x(3,1) : column vector
    See also CROSS2MATRIX
    """
    x = np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    return x

def distPoint2EpipolarLine(F, p1, p2):
    """ Compute the point-to-epipolar-line distance

       Input:
       - F np.ndarray(3,3): Fundamental matrix
       - p1 np.ndarray(3,N): homogeneous coords of the observed points in image 1
       - p2 np.ndarray(3,N): homogeneous coords of the observed points in image 2

       Output:
       - cost: sum of squared distance from points to epipolar lines
               normalized by the number of point coordinates
    """

    N = p1.shape[1]

    homog_points = np.c_[p1, p2]
    epi_lines = np.c_[F.T @ p2, F @ p1]

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2
    cost = np.sqrt( np.sum( np.sum( epi_lines * homog_points, axis = 0)**2 / denom) / N)

    return cost


def invHomMatrix(T):
    """ Inverse of a homogeneous matrix

       Input:
       - T np.ndarray(4,4): homogeneous matrix

       Output:
       - invT np.ndarray(4,4): inverse of T
    """
    R = T[0:3,0:3]
    t = T[0:3,3]
    invT = np.r_[np.c_[R.T, -R.T @ t], np.array([[0,0,0,1]])]
    return invT

def ypr_to_rot_matrix(yaw, pitch, roll):
    """
    Convert Euler angles (ZYX convention) to a rotation matrix.
    
    :param yaw: Yaw angle (rotation around the Z-axis), in radians.
    :param pitch: Pitch angle (rotation around the Y-axis), in radians.
    :param roll: Roll angle (rotation around the X-axis), in radians.
    :return: Rotation matrix.
    """
    rotation_matrix = np.array([
        [np.cos(yaw)*np.cos(pitch), -np.sin(yaw)*np.cos(roll) + np.cos(yaw)*np.sin(pitch)*np.sin(roll), np.sin(yaw)*np.sin(roll) + np.cos(yaw)*np.sin(pitch)*np.cos(roll)],
        [np.sin(yaw)*np.cos(pitch), np.cos(yaw)*np.cos(roll) + np.sin(yaw)*np.sin(pitch)*np.sin(roll), -np.cos(yaw)*np.sin(roll) + np.sin(yaw)*np.sin(pitch)*np.cos(roll)],
        [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
    ])
    return rotation_matrix

def rot_matrix_to_ypr(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles (ZYX convention).
    
    :param rotation_matrix: Rotation matrix.
    :return: Yaw, pitch, and roll angles in radians.
    """
    pitch = -np.arcsin(rotation_matrix[2, 0])
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return yaw, pitch, roll

def fit_plane(points):
    """
    Fit a plane to the given 3D points using the least-squares method.
    Returns the normal vector and the distance to the origin.
    """
    assert len(points) >= 3, "At least 3 points are required."

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, V = np.linalg.svd(centered_points)
    normal_vector = V[-1]
    # Ensure that the normal vector points in the opposite direction of the world coordinate system's z-axis
    if normal_vector[2] > 0:
        normal_vector *= -1
    normal_vector /= np.linalg.norm(normal_vector)
    distance = -np.dot(normal_vector, centroid)
    return normal_vector, distance

def rotation_matrix_to_face(normal_vector, points):
    # assert len(points) >= 2, "At least 2 points are required for line fitting."

    # Construct the design matrix A to represent the line equation ax + by + cz + d = 0
    # X = points[:, 0]  # x-coordinates of the points
    # Y = points[:, 1]  # y-coordinates of the points
    # Z = points[:, 2]  # z-coordinates of the points

    # A = np.vstack((X, Y, Z, np.ones(len(X)))).T

    # # Use SVD to solve the least-squares problem and find the line parameters
    # _, _, Vt = np.linalg.svd(A, full_matrices=False)
    # V = Vt.T
    # params = V[:, -1]

    # # Extract the parameters (a, b, c, d) for the line equation ax + by + cz + d = 0
    # a_opt, b_opt, c_opt, d_opt = params

    # # The direction vector of the line is the normal vector (a, b, c)
    # direction_vector = np.array([a_opt, b_opt, c_opt])

    # # Normalize the direction vector to get the best-fit unit vector
    # direction_vector /= np.linalg.norm(direction_vector)

    # # The line passes through the centroid of the points
    # centroid = np.mean(points, axis=0)

    z_axis = normal_vector
    x_axis = points[0,:] - points[3,:]
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    # y_axis = direction_vector
    # x_axis = np.cross(y_axis, z_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T

    return rotation_matrix


def rotation_matrix_to_align_plane(normal_vector):
    """
    Construct a rotation matrix that aligns the plane's normal vector with the world coordinate system's Z-axis.
    """
    z_axis = np.array([0, 0, 1])  # Z-axis in the world coordinate system
    rotation_axis = np.cross(normal_vector, z_axis)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(normal_vector, z_axis))
    
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])

    rotation_matrix = np.eye(3) + sin_theta * K + (1 - cos_theta) * K @ K
    return rotation_matrix


def MedianFilter(Matrix, Vector):
    """ Matrix: to store Vector, median over the stored matrix elements """
    Matrix[:,:-1] = Matrix[:,1:]
    Matrix[:,-1] = Vector[0:3].reshape(3,)

    if np.all(~np.isnan(Matrix)):
        return np.median(Matrix, axis=1)
    else:
        return Vector[0:3]