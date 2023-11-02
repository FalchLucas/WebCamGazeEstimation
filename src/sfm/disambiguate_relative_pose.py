import numpy as np

from sfm.linear_triangulation import linearTriangulation

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """

    # Projection matrix of camera 1
    M1 = K1 @ np.eye(3,4)

    total_points_in_front_best = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        
        for iSignT in range(2):
            T_C2_C1_test = u3 * (-1)**iSignT
            
            M2 = K2 @ np.c_[R_C2_C1_test, T_C2_C1_test]
            P_C1 = linearTriangulation(points0_h, points1_h, M1, M2)
            
            # project in both cameras
            P_C2 = np.c_[R_C2_C1_test, T_C2_C1_test] @ P_C1
            
            num_points_in_front1 = np.sum(P_C1[2,:] > 0)
            num_points_in_front2 = np.sum(P_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2
                  
            if (total_points_in_front > total_points_in_front_best):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test
                T = T_C2_C1_test
                total_points_in_front_best = total_points_in_front

    return R, T
