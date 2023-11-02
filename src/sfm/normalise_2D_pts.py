import numpy as np

def normalise2DPts(pts):
    """  normalises 2D homogeneous points

     Function translates and normalises a set of 2D homogeneous points
     so that their centroid is at the origin and their mean distance from
     the origin is sqrt(2).

     Usage:   [pts_tilde, T] = normalise2dpts(pts)

     Argument:
       pts -  3xN array of 2D homogeneous coordinates

     Returns:
       pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
       T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    """


    N = pts.shape[1]

    # Convert homogeneous coordinates to Euclidean coordinates (pixels)
    pts_ = pts/pts[2,:]

    # Centroid (Euclidean coordinates)
    mu = np.mean(pts_[:2,:], axis = 1)

    # Average distance or root mean squared distance of centered points
    # It does not matter too much which criterion to use. Both improve the
    # numerical conditioning of the Fundamental matrix estimation problem.
    pts_centered = (pts_[:2,:].T - mu).T

    # Option 1: RMS distance
    sigma = np.sqrt( np.mean( np.sum(pts_centered**2, axis = 0) ) )

    # Option 2: average distance
    # sigma = mean( sqrt(sum(pts_centered.^2)) );

    s = np.sqrt(2) / sigma
    T = np.array([
        [s, 0, -s * mu[0]],
        [0, s, -s * mu[1]],
        [0, 0, 1]])

    pts_tilde = T @ pts_

    return pts_tilde, T


