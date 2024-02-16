import datetime
import logging
import pathlib
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import time
import os
import keyboard
from omegaconf import DictConfig

from plgaze.common import Face, FacePartsName, Visualizer
from plgaze.gaze_estimator import GazeEstimator
from plgaze.utils import get_3d_face_model

class GazeModel:

    def __init__(self, config: DictConfig):
        self.config = config
        self.df = pd.DataFrame()
        self.gaze_estimator = GazeEstimator(config)
        face_model_3d = get_3d_face_model(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera,
                                     face_model_3d.NOSE_INDEX)

        self.output_dir = self._create_output_dir()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

    def get_gaze(self, frame, imshow=False):

        undistorted = cv2.undistort(frame, self.gaze_estimator.camera.camera_matrix,
                                    self.gaze_estimator.camera.dist_coefficients)
        self.visualizer.set_image(frame.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        eye_info = None
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            eye_centers = np.array([0,0,0,0])
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            head_pose_angles = np.array([yaw, pitch, 0])
            head_box = (face.bbox).flatten()[:2]
            right_eye_box = np.array([0,0])
            left_eye_box = np.array([0,0])
            R = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            gaze_vec = R @ face.gaze_vector
            eye_info = {'gaze':gaze_vec, 'EyeRLCenterPos':eye_centers, 'HeadPosAnglesYPR':head_pose_angles, 
                        'HeadPosInFrame':head_box, 'right_eye_box':right_eye_box, 'left_eye_box':left_eye_box, 
                        'EyeState':np.array([1, 1]) }
            if imshow:
                self._draw_gaze_vector(face)
                self._draw_face_bbox(face)
                if self.config.demo.use_camera:
                    self.visualizer.image = self.visualizer.image[:, ::-1]
            
        return eye_info
    
    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
        
    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def run(self, cap) -> None:
        print("Running EyeModel")
        while (cap.isOpened()):
            try:
                ret, frame = cap.read()
            except Exception as e:
                print(f"Could not read from video stream: {e}")
            if ret == False:
                print("Video stream ended")
                break

            eye_info = self.get_gaze(frame=frame, imshow=True)
            if eye_info is None:
                print("No eye info. Eye tracking failed.")
            
            arr = np.array([])
            for i in pd.Series(eye_info).values:
                arr = np.hstack((arr,i))
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.df = pd.concat([ self.df, pd.DataFrame([np.hstack((timestamp, arr))]) ])

            cv2.waitKey(1)
            if keyboard.is_pressed('esc'):
                print("Recording stopped")
                break

        cap.release()
        cv2.destroyAllWindows()
        
        self.df.columns = ['Timestamp', 'gaze_x', 'gaze_y', 'gaze_z', 'REyePos_x', 'REyePos_y', 'LEyePos_x', 'LEyePos_y', 
                           'yaw', 'pitch', 'roll', 'HeadBox_xmin', 'HeadBox_ymin', 'RightEyeBox_xmin', 
                           'RightEyeBox_ymin', 'LeftEyeBox_xmin', 'LeftEyeBox_ymin', 'ROpenClose','LOpenClose']
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.output_dir,'eye_tracking.csv'), index=False)