import numpy as np


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """

    u, _, vh = np.linalg.svd(E)

    # Translation
    u3 = u[:, 2]

    # Rotations
    W = np.array([ [0, -1,  0],
                   [1,  0,  0],
                   [0,  0,  1]])

    R = np.zeros((3,3,2))
    R[:, :, 0] = u @ W @ vh
    R[:, :, 1] = u @ W.T @ vh

    for i in range(2):
        if np.linalg.det(R[:, :, i]) < 0:
            R[:, :, i] *= -1

    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    return R, u3

