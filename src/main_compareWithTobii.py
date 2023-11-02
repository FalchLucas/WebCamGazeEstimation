"""
This script is used to compare the gaze estimation results of the OpenVINO model with the Tobii Eye Tracker 5.
For this script to work, the Tobii Eye Tracker 5 must be connected to the computer and the Tobii Eye Tracker 5 software must be installed and running.
getGaze.exe displays a target on the screen and records the gaze coordiantes of the Tobii Eye Tracker 5.
"""
import os
import cv2
import time
import numpy as np
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel
import subprocess


def main(dir):

    try:     
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        model = EyeModel(dir)

        homtrans = HomTransform(dir)
        # cap=cv2.VideoCapture(0)
        """ for higher resolution (max available: 1920x1080) """
        cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        """ Calibration """
        STransG = homtrans.calibrate(model, cap)

        print("============================")
        print(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

        """ Execute Tobii """
        subprocess.Popen("C:\\temp\\tobiieyetracker5\\build\\Debug\\getGaze.exe", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        homtrans.RecordGaze(model, cap)

    except Exception as e:
        print(f"Something wrong when running EyeModel: {e}")

if __name__ == '__main__':
    dir = "C:\\temp\\WebCamGazeEstimation\\"
    main(dir)