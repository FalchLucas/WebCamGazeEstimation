import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from gaze_tracking.model import EyeModel
import gaze_tracking.gui_opencv as gcv

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation

from sfm.estimate_essential_matrix import estimateEssentialMatrix
from sfm.decompose_essential_matrix import decomposeEssentialMatrix
from sfm.disambiguate_relative_pose import disambiguateRelativePose
from sfm.linear_triangulation import linearTriangulation
from sfm.draw_camera import drawCamera
# from sfm.utils import invHomMatrix, fit_plane, rotation_matrix_to_align_plane
import utilities.utils as util


class SFM():

    def __init__(self, directory) -> None:            
        self.dir = directory
        self.camera_matrix, self.dist_coeffs = gcv.ReadCameraCalibrationData(os.path.join(directory, "camera_data"))
        self.width, self.height, self.width_mm, self.height_mm = gcv.getScreenSize()
        self.S_T_W = np.array([[-1,0,0,self.width/2],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        
    def RunGazeOnScreen(self, model, cap):

        if cap != None:
            out_video, wc_width, wc_height = gcv.get_out_video(cap, os.path.join(self.dir, "results"), file_name = "output_video.mp4", scalewidth=2)

        white_frame = gcv.getWhiteFrame(self.width, self.height)
        target = gcv.Targets(self.width, self.height)
        df = pd.DataFrame()
        frame = None
        frame_prev = None
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except StopIteration:
                break
            
            if frame_prev is None:
                frame_prev = frame
                frame = None

            if frame is not None and frame_prev is not None:
                p1 = model.get_FaceFeatures(frame_prev)
                p2 = model.get_FaceFeatures(frame)
                frame_prev = frame

                E = estimateEssentialMatrix(p1, p2, self.camera_matrix, self.camera_matrix)
                # Extract the relative camera positions (R,T) from the essential matrix
                # Obtain extrinsic parameters (R,t) from E
                Rots, u3 = decomposeEssentialMatrix(E)

                # Disambiguate among the four possible configurations
                G_R_Gp, G_t_Gp = disambiguateRelativePose(Rots, u3, p1, p2, self.camera_matrix, self.camera_matrix)

                # Triangulate a point cloud using the final transformation (R,T)
                M1 = self.camera_matrix @ np.eye(3,4)
                M2 = self.camera_matrix @ np.c_[G_R_Gp, G_t_Gp]
                W_P = linearTriangulation(p1, p2, M1, M2)   # Estimated 3D points of face features in world coordinates (previous frame)

                # World is location of previous frame
                W_T_Gp = np.r_[np.c_[np.eye(3), W_P[0:3,0]], np.array([[0,0,0,1]])]
                Gp_T_G = np.r_[np.c_[G_R_Gp.T, -G_R_Gp.T @ G_t_Gp], np.array([[0,0,0,1]])]
                W_T_G = W_T_Gp @ Gp_T_G

                eye_info = model.get_gaze(frame)
                gaze = eye_info['gaze']

                # Project gaze vector on screen
                Ggaze = self._ProjectVetorOnPlane(util.invHomMatrix(W_T_G), gaze)
                Sgaze = self.S_T_W @ W_T_G @ Ggaze

                EyeState = eye_info['EyeState']
                if np.all(np.array(EyeState) == 1):
                    gazeframe, SetPos = target.DrawTargetGaze(white_frame, Sgaze[0:3])
                    if out_video is not None:
                        final_frame = np.concatenate((cv2.flip(cv2.resize(gazeframe, (wc_width, wc_height)), 1), frame), axis=1)
                        out_video.write(final_frame)

                else:
                    # return np.array([-10,-10,-10])
                    pass
            
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, eye_info['gaze'], Sgaze[0:3].reshape(-1), W_P[0:3,0].reshape(-1), G_t_Gp)) ]) ])

            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        df.columns = ['timestamp(hh:m:s.ms)','gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x', 'Sgaze_y', 'Sgaze_z', 'W_Gpx', 'W_Gpy', 'W_Gpz', 'G_t_Gpx', 'G_t_Gpy', 'G_t_Gpz']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "GazeTracking.csv"))

    def getReferenceFrame(self, video_path):
        video_path = os.path.join(self.dir, video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            exit()

        accum_frame = None
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if accum_frame is None:
                # Initialize the accumulator with the first frame
                accum_frame = np.zeros_like(frame, dtype=np.float32)

            accum_frame += frame.astype(np.float32)
            frame_count += 1

        cap.release()

        if frame_count > 0:
            # Compute the average frame
            average_frame = (accum_frame / frame_count).astype(np.uint8)
            self.average_frame = average_frame

            return average_frame
        
        else:
            print("Error: No frames in video")
            return None


    def sfm_video(self, model, video_path):
        video_path = os.path.join(self.dir, video_path)
        cap = cv2.VideoCapture(video_path)

        out_video,_,_ = gcv.get_out_video(cap, os.path.join(self.dir, "results"), file_name = "eye_features.mp4", scalewidth=2)

        frame = None
        # frame_prev = self.average_frame
        frame_prev = None
        W_P = None

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        plots = []
        df = pd.DataFrame()
        dfT = pd.DataFrame() 
        while cap.isOpened():            
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"Could not read from video stream: {e}")

            if ret == False:
                break

            if frame_prev is not None:
                # W_P is defined in previous frame
                # W_T_G1, W_T_G2, W_P = self.get_GazeToWorld(model, frame, frame_prev)
                W_T_G1, W_T_G2, W_P = self.get_GazeToWorld(model, frame_prev, frame)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, W_P.flatten(), np.median(W_P, axis=0))) ]) ])
                dfT = pd.concat([ dfT, pd.DataFrame(W_T_G1) ])

                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(W_P[:,0], W_P[:,1], W_P[:,2], marker = 'o')
                drawCamera(ax, W_T_G1[:3,3], W_T_G1[:3,:3], length_scale = 0.1)
                # drawCamera(ax, W_T_G2[:3,3], W_T_G2[:3,:3], length_scale = 0.1)
                drawCamera(ax, np.zeros(3), np.eye(3), length_scale = 0.1)
                ax.text(-0.1,-0.1,-0.1,"W")
                # ax.view_init(elev=80, azim=0)
                            
                plots.append([ax])
                # plt.pause(0.001)

            frame_prev = frame.copy()
                
            key_pressed = cv2.waitKey(1)
            if key_pressed == 27:
                # exit()
                break

            if W_P is not None:                
                for p in W_P:
                    I_P = self.camera_matrix @ p.reshape(3,1)
                    I_P = I_P / I_P[2]
                    WtG = self.camera_matrix @ W_T_G1[:3,3]
                    WtG = WtG / WtG[2]
                    x_axis = self.camera_matrix @ (W_T_G1[:3,3]+0.1*W_T_G1[:3,0])
                    x_axis = x_axis / x_axis[2]
                    y_axis = self.camera_matrix @ (W_T_G1[:3,3]+0.1*W_T_G1[:3,1])
                    y_axis = y_axis / y_axis[2]
                    z_axis = self.camera_matrix @ (W_T_G1[:3,3]+0.1*W_T_G1[:3,2])
                    z_axis = z_axis / z_axis[2]
                    cv2.drawMarker(frame, tuple(I_P.astype(int)[0:2].flatten()), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2) 
                    cv2.drawMarker(frame, tuple(WtG.astype(int)[0:2].flatten()), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=3) 
                    cv2.line(frame, tuple(WtG.astype(int)[0:2].flatten()), tuple((x_axis).astype(int)[0:2].flatten()), color=(0,0,255), thickness=2)
                    cv2.line(frame, tuple(WtG.astype(int)[0:2].flatten()), tuple((y_axis).astype(int)[0:2].flatten()), color=(0,255,0), thickness=2)
                    cv2.line(frame, tuple(WtG.astype(int)[0:2].flatten()), tuple((z_axis).astype(int)[0:2].flatten()), color=(255,0,0), thickness=2)


            window_name = "CalibWindow"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(window_name, frame)

            # draw_frame = frame_prev.copy()
            draw_frame = frame.copy()
            p1 = model.get_FaceFeatures(draw_frame)
            # for idx, p in enumerate(p1.T):
            #     cv2.drawMarker(draw_frame, tuple(p.astype(int)[0:2].flatten()), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
            #     cv2.putText(draw_frame, str(idx), tuple(p.astype(int)[0:2].flatten()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            if out_video is not None:      
                # out_video.write(frame)
                out_video.write(cv2.hconcat([draw_frame, frame]))
            else:
                print("No output video")

        
        cap.release()
        cv2.destroyAllWindows()
        ani = animation.ArtistAnimation(fig, plots)
        ani.save(os.path.join(self.dir, "results", 'animation.mp4'), writer='ffmpeg')
        
        df.columns = ['timestamp(hh:m:s.ms)'] + ['W_Px', 'W_Py', 'W_Pz']*W_P.shape[0] + ['W_Px_mean', 'W_Py_mean', 'W_Pz_mean']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "W_P.csv"))
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani = animation.ArtistAnimation(fig, plots, interval=50, blit=True)
        # ani.save(os.path.join(self.dir, "results", 'animation.mp4'), writer=writer)
        dfT.to_csv(os.path.join(self.dir, "results", "W_T_G.csv"), index=False)

        return dfT.to_numpy()

    def sfm_image(self, model, image_path):
        image_path = os.path.join(self.dir, image_path)
        img_1 = np.array(cv2.imread(os.path.join(image_path, "im1.jpg")))
        img_2 = np.array(cv2.imread(os.path.join(image_path, "im2.jpg")))

        p1 = model.get_FaceFeatures(img_1)[:2,:]
        p1 = cv2.undistortPoints(p1, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)
        p2 = model.get_FaceFeatures(img_2)[:2,:]
        p2 = cv2.undistortPoints(p2, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)

        E = cv2.findEssentialMat(p1, p2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
        _, G2_R_G1, G2_t_G1, _= cv2.recoverPose(E, p1, p2, self.camera_matrix)    # C1 cosy is world coordinate system

        M1 = self.camera_matrix @ np.eye(3,4)
        M2 = self.camera_matrix @ np.c_[G2_R_G1, G2_t_G1]
        points_4d_homogeneous = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
        W_P = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1,3)   # 35x3
        W_P = W_P/np.linalg.norm(W_P, axis=1)[:,np.newaxis]
        W_P[W_P[:,2]<0] = W_P[W_P[:,2]<0]*(-1)

        W_T_G1 = np.r_[np.c_[np.array([[1,0,0],[0,-1,0],[0,0,-1]]), np.mean(np.array([W_P[0,:],W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]  

        G1_T_G2 = np.r_[np.c_[G2_R_G1.T, -G2_R_G1.T @ G2_t_G1], np.array([[0,0,0,1]])]
        W_T_G2 = W_T_G1 @ G1_T_G2       # not really useful

        # Visualize the 3-D scene
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(W_P[:,0], W_P[:,1], W_P[:,2], marker = 'o')
        drawCamera(ax, W_T_G1[:3,3], W_T_G1[:3,:3], length_scale = 0.1)
        drawCamera(ax, W_T_G2[:3,3], W_T_G2[:3,:3], length_scale = 0.1)
        drawCamera(ax, np.zeros(3), np.eye(3), length_scale = 0.1)
        ax.text(-0.1,-0.1,-0.1,"W")

        plt.show()

    def get_GazeToWorld(self, model, frame_prev, frame):
        p1 = model.get_FaceFeatures(frame_prev)[:2,:]
        p1 = cv2.undistortPoints(p1, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)
        p2 = model.get_FaceFeatures(frame)[:2,:]
        p2 = cv2.undistortPoints(p2, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix).reshape(-1,2)

        E = cv2.findEssentialMat(p1, p2, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)[0]
        _, G2_R_G1, G2_t_G1, _= cv2.recoverPose(E, p1, p2, self.camera_matrix)    # G1 cosy is world coordinate system

        # Triangulate a point cloud using the final transformation (R,T)
        M1 = self.camera_matrix @ np.eye(3,4)
        M2 = self.camera_matrix @ np.c_[G2_R_G1, G2_t_G1]
        points_4d_homogeneous = cv2.triangulatePoints(M1, M2, p1.T, p2.T)
        W_P = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1,3)   # 35x3

        W_P = W_P/np.linalg.norm(W_P, axis=1)[:,np.newaxis]
        W_P[W_P[:,2]<0] = W_P[W_P[:,2]<0]*(-1)      # if z<0, change sign of x,y,z

        # rotation of face in 3d, however provided gaze vector is already rotated back so it aligns with world coordinate system
        normal_vector,_ = util.fit_plane(W_P)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)
        W_R_G1 = util.rotation_matrix_to_face(normal_vector, np.array([W_P[0,:],W_P[2,:],W_P[3,:],W_P[18,:]]) )
        # print(f"W_R_G1\n{np.array2string(W_R_G1, formatter={'float': lambda x: f'{x:.2f}'})}")

        # World is location of previous frame
        WRotG = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        W_T_G1 = np.r_[np.c_[WRotG, np.mean(np.array([W_P[0,:],W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]
        # W_T_G1 = np.r_[np.c_[W_R_G1, np.mean(np.array([W_P[0,:],W_P[2,:]]), axis=0)[:,None]], np.array([[0,0,0,1]])]  
        G1_T_G2 = np.r_[np.c_[G2_R_G1.T, -G2_R_G1.T @ G2_t_G1], np.array([[0,0,0,1]])]
        W_T_G2 = W_T_G1 @ G1_T_G2       # not really useful
        if W_T_G2[2,3]<0:
            W_T_G2[:3,3] = W_T_G2[:3,3]*(-1)

        # W_T_G1[:2,3] = W_T_G1[:2,3]*(-1)    # flip x,y axis

        return W_T_G1, W_T_G2, W_P

    def _ProjectVetorOnPlane(self, Trans, vector):
        """ Translation of homogenous Trans-Matrix must be in same coordinate system as Vector """
        vector = vector.reshape(3,1)
        VectorNormal2Plane = (Trans[0:3,0:3] @ np.array([[0],[0],[1]]))
        transVec = Trans[0:3,3]
        t = (VectorNormal2Plane.T @ transVec) / (VectorNormal2Plane.T @ vector)
        Vector2Plane = np.vstack((t*vector, 1))
        return Vector2Plane

if __name__ == '__main__':
    dir = "C:\\tmp\\GazeEstimation\\"
    model = EyeModel(dir)
    sfm = SFM(dir)

    # video_path = os.path.join(dir, "results", "calibrate.mp4")
    # average_frame = sfm.getReferenceFrame(video_path)
    # p1 = model.get_FaceFeatures(average_frame)
    # for idx, p in enumerate(p1.T):
    #     cv2.drawMarker(average_frame, tuple(p.astype(int)[0:2].flatten()), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
    #     cv2.putText(average_frame, str(idx), tuple(p.astype(int)[0:2].flatten()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the average frame
    # cv2.imshow("Average Frame", average_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    video_path = os.path.join(dir, "results", "output_video.mp4")
    sfm.sfm_video(model, video_path)


    # cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # # cap.set(cv2.CAP_PROP_SETTINGS, 1)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # sfm.RunGazeOnScreen(model, cap)

    # image_path = os.path.join(dir, "results")
    # sfm.sfm_image(model, image_path)