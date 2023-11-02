import numpy as np

def fundamentalEightPoint(p1, p2):
    """ The 8-point algorithm for the estimation of the fundamental matrix F

     The eight-point algorithm for the fundamental matrix with a posteriori
     enforcement of the singularity constraint (det(F)=0).
     Does not include data normalization.

     Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """

    # Sanity checks
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    
    num_points = p1.shape[1]
    assert(num_points>=8), \
            'Insufficient number of points to compute fundamental matrix (need >=8)'

    # Compute the measurement matrix A of the linear homogeneous system whose
    # solution is the vector representing the fundamental matrix.
    A = np.zeros((num_points,9))
    for i in range(num_points):
        A[i,:] = np.kron( p1[:,i], p2[:,i] ).T
    
    # "Solve" the linear homogeneous system of equations A*f = 0.
    # The correspondences x1,x2 are exact <=> rank(A)=8 -> there exist an exact solution
    # If measurements are noisy, then rank(A)=9 => there is no exact solution, 
    # seek a least-squares solution.
    _, _, vh= np.linalg.svd(A,full_matrices = False)
    F = np.reshape(vh[-1,:], (3,3)).T

    # Enforce det(F)=0 by projecting F onto the set of 3x3 singular matrices
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = u @ np.diag(s) @ vh

    return F

