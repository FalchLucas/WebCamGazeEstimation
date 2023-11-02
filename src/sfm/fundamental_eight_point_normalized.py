import numpy as np

from .fundamental_eight_point import fundamentalEightPoint
from .normalise_2D_pts import normalise2DPts

def fundamentalEightPointNormalized(p1, p2):
    """ Normalized Version of the 8 Point algorith
     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """

    # Normalize each set of points so that the origin
    # is at centroid and mean distance from origin is sqrt(2).
    p1_tilde, T1 = normalise2DPts(p1)
    p2_tilde, T2 = normalise2DPts(p2)

    # Linear solution
    F = fundamentalEightPoint(p1_tilde, p2_tilde)

    # Undo the normalization
    F = T2.T @ F @ T1

    return F

