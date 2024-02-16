import scipy.optimize as opt
import cv2
import os
import keyboard
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gaze_tracking.gui_opencv as gcv
from utilities.kalman import Kalman
from sfm.sfm_module import SFM
import utilities.utils as util


class HomTransform:
    """
    Calibration from gaze coordinates to screen coordinates
    """
    def __init__(self, directory) -> None:            
        self.dir = directory
        self.width, self.height, self.width_mm, self.height_mm = gcv.getScreenSize()
        self.df = pd.DataFrame()
        self.sfm = SFM(directory)
        self.camera_matrix, self.dist_coeffs = gcv.ReadCameraCalibrationData(os.path.join(directory, "camera_data"))
        self.inv_camera_matrix = np.linalg.inv(self.camera_matrix)

    def RecordGaze(self, model, cap, sfm=False):
        df = pd.DataFrame()
        frame_prev = None
        WTransG1 = np.eye(4)
        while cap.isOpened():
            try:
                ret, frame = cap.read()
            except StopIteration:
                break

            eye_info = model.get_gaze(frame)
            gaze = eye_info['gaze']

            if sfm:
                if frame_prev is not None:                
                    WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)        # WtG1 is a unit vector, has to be scaled            
                frame_prev = frame
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
            else:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)

            FSgaze = self._mm2pixel(FSgaze)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, eye_info['gaze'], FSgaze.flatten(), eye_info['EyeRLCenterPos'], eye_info['HeadPosAnglesYPR'], eye_info['HeadPosInFrame'])) ]) ])

            cv2.waitKey(1)
            if keyboard.is_pressed('esc'):
                print("Recording stopped")
                break
        cap.release()
        df.columns = ['timestamp(hh:m:s.ms)','gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x', 'Sgaze_y', 'Sgaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadPos_x', 'HeadPos_y']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "MyGazeTracking.csv"))

    def RunGazeOnScreen(self, model, cap, sfm=False):
        """ Present different trajectories on screen and record gaze
        """

        if cap != None:
            out_video, wc_width, wc_height = gcv.get_out_video(cap, os.path.join(self.dir, "results"), file_name = "output_video.mp4", scalewidth=2)

        white_frame = gcv.getWhiteFrame(self.width, self.height)
        df = pd.DataFrame()
        target = gcv.Targets(self.width, self.height)
        frame_prev = None
        WTransG1 = np.eye(4)
        target.setSetPos([int(self.width/8), int(self.height/8)])   # for DrawSpecificTarget()
        FSgaze = np.array([[-10],[-10],[-10]])
        while cap.isOpened():
            
            # gazeframe, SetPos = target.DrawTargetGaze(white_frame, self._mm2pixel(FSgaze))
            # gazeframe, SetPos = target.DrawRectangularTargets(white_frame, self._mm2pixel(FSgaze))
            # gazeframe, SetPos = target.DrawSingleTargets(white_frame, self._mm2pixel(FSgaze))
            gazeframe, SetPos = target.DrawTargetInMiddle(white_frame, self._mm2pixel(FSgaze))

            try:
                ret, frame = cap.read()
            except StopIteration:
                break
            
            # gray_image, prediction, morphedMask, falseColor, centroid = model.get_iris_Cnn(frame)
            # Undistort the image
            # frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            eye_info = model.get_gaze(frame)

            gaze = eye_info['gaze']

            if frame_prev is not None and sfm:                
                WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame)        # WtG1 is a unit vector, has to be scaled   

            frame_prev = frame

            if sfm:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen_sfm(gaze, WTransG1)
            else:
                FSgaze, Sgaze, Sgaze2 = self._getGazeOnScreen(gaze)
            
            SetPos = self._pixel2mm(SetPos)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            df = pd.concat([ df, pd.DataFrame([np.hstack((timestamp, eye_info['gaze'], FSgaze.flatten(), eye_info['EyeRLCenterPos'], 
                                                        eye_info['HeadPosAnglesYPR'], eye_info['HeadPosInFrame'], SetPos, 0, WTransG1[:3,3].flatten(), 
                                                        Sgaze.flatten(), Sgaze2.flatten(), eye_info['EyeState'] )) ]) ])

            if out_video is not None:
                final_frame = np.concatenate((cv2.flip(cv2.resize(gazeframe, (wc_width, wc_height)), 1), frame), axis=1)
                out_video.write(final_frame)
                # out_video.write(frame)

            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()
        df.columns = ['time_sec','gaze_x', 'gaze_y', 'gaze_z', 'Sgaze_x', 'Sgaze_y', 'Sgaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y',
                      'yaw', 'pitch', 'roll', 'HeadPos_x', 'HeadPos_y', 'set_x', 'set_y', 'set_z', 'WTransG_x', 'WTransG_y', 'WTransG_z', 
                      'RegSgaze_x', 'RegSgaze_y', 'RegSgaze_z', 'CalPSgaze_x', 'CalPSgaze_y', 'CalPSgaze_z', 'ROpenClose','LOpenClose']
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(self.dir, "results", "GazeTracking.csv"))

        return

    def calibrate(self, model, cap, sfm=False):
        frame = gcv.getWhiteFrame(self.width, self.height)
        if cap != None:
            out_video,_,_ = gcv.get_out_video(cap, os.path.join(self.dir, "results"))
            self.WC_width, self.WC_height = gcv.getWebcamSize(cap)

        target = gcv.Targets(self.width, self.height)
        frame_prev = None
        WTransG1 = np.zeros((4,4))
        target.tstart = time.time()
        while (cap.isOpened()):
            idx, SetPos = target.getTargetCalibration()
            if idx == None:
                break
            
            """ Draw Target on white frame """
            frame2 = frame.copy()
            cv2.circle(frame2, tuple(SetPos), 15, (0, 0, 255), -1)
            length = 25
            thickness = 4
            if idx == 9:
                cv2.arrowedLine(frame2, (SetPos[0]+length, SetPos[1]), (SetPos[0]-length, SetPos[1]), (0, 0, 0), thickness)
            if idx == 10:                
                cv2.arrowedLine(frame2, (SetPos[0]-length, SetPos[1]), (SetPos[0]+length, SetPos[1]), (0, 0, 0), thickness)
            if idx == 11:
                cv2.arrowedLine(frame2, (SetPos[0], SetPos[1]-length), (SetPos[0], SetPos[1]+length), (0, 0, 0), thickness)
            if idx == 12:
                cv2.arrowedLine(frame2, (SetPos[0], SetPos[1]+length), (SetPos[0], SetPos[1]-length), (0, 0, 0), thickness)

            gcv.display_window(frame2)

            try:
                ret, frame_cam = cap.read()
            except Exception as e:
                print(f"Could not read from video stream: {e}")
            if ret == False:
                print("Video stream ended")
                break
            
            if frame_prev is not None and sfm:
                WTransG1, WTransG2, W_P = self.sfm.get_GazeToWorld(model, frame_prev, frame_cam)

            frame_prev = frame_cam.copy()

            # frame_cam = cv2.undistort(frame_cam, self.camera_matrix, self.dist_coeffs)
            eye_info = model.get_gaze(frame=frame_cam, imshow=False)
            if eye_info is None:
                raise Exception("No eye info. Eye tracking failed.")
            
            arr = np.array([])
            for i in pd.Series(eye_info).values:
                arr = np.hstack((arr,i))
            timestamp = time.time_ns()/1000000000
            SetPos = self._pixel2mm(SetPos)
            self.df = pd.concat([ self.df, pd.DataFrame([np.hstack((timestamp, idx, arr, SetPos, 0, WTransG1.flatten()))]) ])

            if out_video is not None:      
                out_video.write(frame_cam)
            else:
                print("No output video")

            key_pressed = cv2.waitKey(1)   # this is needed
            if key_pressed == 27:
                exit()

        cv2.destroyAllWindows()
        out_video.release()

        self.df.columns = ['Timestamp', 'idx', 'gaze_x', 'gaze_y', 'gaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 'yaw', 'pitch', 'roll', 'HeadBox_xmin', 'HeadBox_ymin', 'RightEyeBox_xmin', 'RightEyeBox_ymin', 'LeftEyeBox_xmin', 'LeftEyeBox_ymin', 'ROpenClose','LOpenClose', 'set_x', 'set_y', 'set_z'] + 16*['WTransG']
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.dir, "results", "Calibration.csv"))

        gaze, SetVal, WTransG, g = self._RemoveOutliers()
    
        if sfm:
            STransW, scaleWtG, STransG = self._fitSTransG_sfm(gaze, SetVal, WTransG, g)
        else:
            STransG = self._fitSTransG(gaze, SetVal, g)

        Sg, SgCalib = self._getCalibValuesOnScreen(g, STransG)
        """ Plot Gaze On Screen"""
        self._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")
        self._WriteStatsInFile(STransG)

        return STransG

    def _getGazeOnScreen(self, gaze):
        scaleGaze = self._getScale(gaze, self.STransG)
        Sgaze = (self.STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """   
        for i in range(len(self.StG)):
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_                                

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)

        """
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system
        Sgaze2 = gaze vector from calibration point
        """
        return FSgaze, Sgaze, Sgaze2

    def _getGazeOnScreen_sfm(self, gaze, WTransG):
        WTransG[:3,3] = self.scaleWtG*WTransG[:3,3]
        STransG = self.STransW @ WTransG
        scaleGaze = self._getScale(gaze, STransG)
        Sgaze = (STransG @ np.vstack((scaleGaze*gaze[:,None], 1)))[:3]

        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        dist = np.inf            
        """ Compute STransG for all calibration points and choose the one with the smallest distance to the overall gaze point on screen """   
        for i in range(len(self.StW)):
            STransG_ = np.vstack((np.hstack((SRotW, self.StW[i].reshape(3,1))), np.array([0,0,0,1]))) @ WTransG
            scaleGaze = self._getScale(gaze, STransG_)
            Sgaze_ = (STransG_ @ np.vstack((scaleGaze*gaze[:,None],1)))[0:3]
            if np.linalg.norm(Sgaze - Sgaze_) < dist:
                dist = np.linalg.norm(Sgaze - Sgaze_)
                Sgaze2 = Sgaze_

        FSgaze = np.median(np.hstack((Sgaze, Sgaze2)), axis=1).reshape(3,1)
        """
        FSgaze = fused gaze vector, overall and for each calibration point
        Sgaze = overall gaze vector, determined over regression in screen coordinate system with head movement
        Sgaze2 = gaze vector from calibration point with head movements
        """
        return FSgaze, Sgaze, Sgaze2

    def _fitSTransG(self, gaze, SetVal, g):
        
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 

        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        gaze = gaze[:,:,None]

        """ Without sfm """
        def alignError(x, *const):
            SRotG, gaze, SetVal = const
            StG = np.array([[x[0]],[x[1]],[x[2]]])
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-SRotG.T @ StG))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + StG
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()
        
        const = (SRotG, gaze, SetVal)
        x0 = np.array([self.width/2, self.height/2, self.width])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StG = np.array([[xopt[0]],[xopt[1]],[xopt[2]]])
        STransG = np.r_[np.c_[SRotG, StG], np.array([[0,0,0,1]])]

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]

        self.STransG = STransG

        return STransG
    
    def _fitSTransG_sfm(self, gaze, SetVal, WTransG, g):
        gaze = gaze.to_numpy()
        SetVal = SetVal.to_numpy() 
        WTransG = WTransG.to_numpy().reshape(-1,4,4)

        WRotG = WTransG[:,:3,:3]
        WtG = WTransG[:,:3,3]
        SRotW = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])

        gaze = gaze[:,:,None]

        """ Model over camera coordinate system getting gaze from SFM  """
        def alignError(x, *const):
            SRotW, WRotG, gaze, WtG, SetVal = const
            StW = np.array([[x[1]],[x[2]],[0]])
            SRotG = SRotW @ WRotG
            Gz = np.array([[0],[0],[1]])
            mu = (Gz.T @ (-np.transpose(SRotG, axes=(0,2,1)) @ (SRotW @ (x[0]*WtG[:,:,None]) + StW)))/(Gz.T @ gaze)
            Sg = SRotG @ (mu*gaze) + SRotW @  (x[0]*WtG[:,:,None]) + StW
            error = SetVal[:,:,None] - Sg   # (87x3x1)
            return error.flatten()

        const = (SRotW, WRotG, gaze, WtG, SetVal)
        x0 = np.array([1, self.width/2, self.height/2])
        res = opt.least_squares(alignError, x0, args=const)
        print(f"res.optimality = {res.optimality}")
        xopt = res.x
        print(f"x_optim = {xopt}")
        StW = np.array([[xopt[1]],[xopt[2]],[0]])
        self.STransW = np.r_[np.c_[SRotW, StW], np.array([[0,0,0,1]])]
        WTransG = np.concatenate((np.c_[WRotG, xopt[0]*WtG[:,:,None]], np.tile(np.array([[0, 0, 0, 1]]), (WtG.shape[0], 1, 1))), axis=1)
        STransG = self.STransW @ np.median(WTransG, axis=0)
        self.scaleWtG = xopt[0]

        WtG = np.median(WtG[:,:,None], axis=0)

        """ Transformation Matrix to Auxiliary points """
        size = len(g)
        self.StW = [None]*size
        self.StG = [None]*size
        for i in range(size):
            scaleGaze = self._getScale(np.median(g[i],axis=0), STransG)     # compute scale for gaze vector for each calibration point
            STransG_, GTransS_ = self._getSTransG(SRotG, self.SetValues[i], np.median(g[i],axis=0), scaleGaze)
            self.StG[i] = STransG_[:3,3,None]
            self.StW[i] = STransG_[:3,3,None] - SRotW @ (self.scaleWtG*WtG)

        self.STransG = STransG

        return self.STransW, self.scaleWtG, STransG
        
    def _getCalibValuesOnScreen(self, g, STransG):
        Sg = [None]*len(g)
        SgCalib = [None]*len(g)
        # SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        SRotG = STransG[:3,:3]
        for i in range(len(g)):
            gaze = g[i].to_numpy()
            scaleGaze = self._getScale(gaze, STransG)
            Sg[i] = (STransG @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]
            STransG_ = np.vstack((np.hstack((SRotG,self.StG[i].reshape(3,1))), np.array([0,0,0,1])))
            scaleGaze = self._getScale(gaze, STransG_)
            SgCalib[i] = (STransG_ @ np.concatenate(( (scaleGaze*gaze[:,:,None]), np.ones((gaze.shape[0],1,1))), axis=1))[:,:3,:]

        return Sg, SgCalib

    def _getSTransG(self, SRotG, SposA, gazeVector, scaleGaze):
        STransA = np.vstack((np.hstack((np.eye(3), SposA)), np.array([0,0,0,1])))      
        ATransG = np.vstack((np.hstack((SRotG, -SRotG.T @ (scaleGaze*gazeVector[:,None]))), np.array([0,0,0,1])))
        STransG = STransA @ ATransG
        GTransS = np.vstack((np.hstack((STransG[0:3,0:3].T, -STransG[0:3,0:3].T @ STransG[0:3,3].reshape(3,1))), np.array([0,0,0,1])))

        return STransG, GTransS

    def _getScale(self, gaze, STransG):
        Gz = np.array([[0],[0],[1]])
        GTransS = util.invHomMatrix(STransG)
        GtS = GTransS[:3,3].reshape(3,1)
        if np.ndim(gaze) == 1:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,None])
        elif np.ndim(gaze) == 2:
            scaleGaze = (Gz.T @ GtS) / (Gz.T @ gaze[:,:,None])

        return scaleGaze

    def _ProjectVetorOnPlane(self, Trans, vector):
        """ Translation of homogenous Trans-Matrix must be in same coordinate system as Vector """
        vector = vector.reshape(3,1)
        # VectorNormal2Plane = (Trans @ np.array([[0],[0],[1],[1]]))[0:3]
        VectorNormal2Plane = (Trans[:3,:3] @ np.array([[0],[0],[1]]))
        # Gz = self.GTransB[0:3,2].reshape(3,1) # not sure why this would work for Tobii (was implemented before)
        transVec = Trans[:3,3]
        t = (VectorNormal2Plane.T @ transVec) / (VectorNormal2Plane.T @ vector)
        Vector2Plane = np.vstack((t*vector, 1))
        return Vector2Plane

    def _RemoveOutliers(self):
        """ Remove Outliers """
        idx = int(pd.unique(self.df['idx'])[-1])+1  # if head turning use -3 otherwise +1
        g = [None]*idx
        s = [None]*idx
        WTG = [None]*idx
        for i in range(idx):            
            g_ = self.df[self.df['idx'].values==i].loc[:,'gaze_x':'gaze_z']
            # sign = np.sign(np.median(g_, axis=0)[0])
            set_val = self.df[self.df['idx'].values==i].loc[:,'set_x':'set_z']
            WTG_ = self.df[self.df['idx'].values==i].filter(like='WTransG')
            mask = self._MaskOutliers(g_.loc[:,'gaze_x']) & self._MaskOutliers(g_.loc[:,'gaze_y']) #& (sign*g_.loc[:,'gaze_x'] > 0)          
            g[i] = g_[mask]
            s[i] = set_val[mask]
            WTG[i] = WTG_[mask]
        
        self.SetValues = [v.to_numpy()[0][:,None] for v in s]
        gaze = pd.concat(g, axis=0)
        SetVal = pd.concat(s, axis=0)
        W_T_G = pd.concat(WTG, axis=0)

        return gaze, SetVal, W_T_G, g

    def _MaskOutliers(self, arr, std_threshold=1):
        """
        Removes outliers from a NumPy array using the standard deviation method.
        Parameters:
            arr (numpy.ndarray): The input array.
            std_threshold (float): The number of standard deviations from the mean to use as the threshold for outlier detection.
        Returns:
            numpy.ndarray: The mask to remove outliers.
        """
        mean = np.mean(arr)
        std = np.std(arr)
        threshold = std_threshold * std
        mask = np.abs(arr - mean) < threshold
        return mask

    def _MaskOutliersPercentile(self, array):
        q75,q25 = np.percentile(array,[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        return (array > min) & (array < max)

    def _WriteStatsInFile(self, STransG):
        """ Write stats in file """
        SRotG = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        with open(os.path.join(self.dir, "results", 'stats.txt'), 'w') as f:
            f.write(f"Transformation matrices: \n")
            f.write(f"STransG1\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[0].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG2\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[1].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG3\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[2].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG4\n{np.array2string(np.vstack((np.hstack((SRotG,self.StG[3].reshape(3,1))), np.array([0,0,0,1]))), formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.3f}'})}\n")
            f.write(f"Screen Information: \n")
            f.write(f"Width: {self.width}px, {self.width_mm}mm\n")
            f.write(f"Height: {self.height}px, {self.height_mm}mm\n")
            f.write(f"Webcam Information: \n")
            f.write(f"Width: {self.WC_width}px\n")
            f.write(f"Height: {self.WC_height}px\n")

    def _getARotG(self, p_origin, p_xCoord, p_yCoord):
        """ Rotation Matrix """
        GxA = p_xCoord - p_origin
        GxA = GxA/np.linalg.norm(GxA)
        GyA = p_yCoord - p_origin
        GyA = GyA/np.linalg.norm(GyA)
        GzA = self._cross(GxA, GyA)
        GRotA = np.hstack((GxA.reshape(3,1), GyA.reshape(3,1), GzA.reshape(3,1)))
        ARotG = GRotA.transpose()

        return ARotG

    def _mm2pixel(self, vector_mm):
        vector = vector_mm.copy()
        if vector.ndim == 2 and vector.shape[0] == 3:
            vector[0] = int(vector[0] * self.width/self.width_mm)
            vector[1] = int(vector[1] * self.height/self.height_mm)
            vector[2] = int(vector[2])
        elif vector.ndim == 3 and vector.shape[1] == 3:
            vector[:,0] = (vector[:,0] * self.width/self.width_mm).astype(int)
            vector[:,1] = (vector[:,1] * self.height/self.height_mm).astype(int)
            vector[:,2] = (vector[:,2]).astype(int)
        else:
            raise Exception("Vector has wrong shape")

        return vector

    def _pixel2mm(self, vector_px):
        if isinstance(vector_px, list):
            vector_px = np.array(vector_px)
        vector = vector_px.copy()
        if vector.ndim == 1 and vector.shape[0] == 2:
            vector[0] = vector[0] * self.width_mm/self.width
            vector[1] = vector[1] * self.height_mm/self.height
        elif vector.ndim == 2 and vector.shape[1] == 2:
            vector[:,0] = vector[:,0] * self.width_mm/self.width
            vector[:,1] = vector[:,1] * self.height_mm/self.height
        else:
            raise Exception("Vector has wrong shape")

        return vector

    def _PlotGaze2D(self, g, Sg, SgCalib, name="GazeOnScreen"):

        # Sg1 = self._mm2pixel(Sg1)
        # Sg2 = self._mm2pixel(Sg2)
        # Sg3 = self._mm2pixel(Sg3)
        # Sg4 = self._mm2pixel(Sg4)
        # SetBp1 = self._mm2pixel(self.SetValues[0])
        # SetBp2 = self._mm2pixel(self.SetValues[1])
        # SetBp3 = self._mm2pixel(self.SetValues[2])
        # SetBp4 = self._mm2pixel(self.SetValues[3])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

        legend = [None]*len(g)
        for i in range(len(g)):
            """ Axis 0: Raw gaze points """
            gaze = g[i].to_numpy()
            ax[0].scatter(gaze[:,0],gaze[:,1])            
            legend[i] = f"p{i+1} values"
            """ Axis 1: Gaze on screen """
            ax[1].scatter(Sg[i][:,0],Sg[i][:,1])

        for i in range(len(g)):
            gaze = g[i].to_numpy()
            ax[0].plot(np.median(gaze[:,0]),np.median(gaze[:,1]),'r+', linewidth=4,  markersize=12)
            ax[1].plot(np.median(Sg[i][:,0]),np.median(Sg[i][:,1]),'r+', linewidth=4,  markersize=12)
            # ax[1].plot(np.median(SgCalib[i][:,0]),np.median(SgCalib[i][:,1]),'k+', linewidth=4,  markersize=12)
            ax[1].plot(self.SetValues[i][0],self.SetValues[i][1],'y*', linewidth=4, markersize=12)


        # ax[0].legend(legend+["Median gaze point"])
        ax[0].set_title('x-y-corrdinates of raw unit gaze points')
        ax[0].set_xlabel("x-direction (unit length)")
        ax[0].set_ylabel("y-direction (unit length)")
        ax[0].grid()
        # ax[1].legend(legend+["Median gaze point", "Displayed Point"])
        ax[1].set_xlabel("x-direction (mm)")
        ax[1].set_ylabel("y-direction (mm)")
        # ax[1].set_title(f"Gaze on screen with resolution {self.width}x{self.height}")
        ax[1].set_title(f"Gaze on screen with dimensions {self.width_mm}mmx{self.height_mm}mm")
        ax[1].grid()

        plt.savefig(os.path.join(self.dir, "results", name))


if __name__ == '__main__':
    print("Noting called from main")