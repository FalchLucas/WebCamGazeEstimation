import os
import cv2
import time
import numpy as np
from gaze_tracking.homtransform import HomTransform
from gaze_tracking.model import EyeModel

def main(dir):

    try:                        
        output_directory = os.path.join(dir, "results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        start_model_load_time = time.time()
        model = EyeModel(dir)
        total_model_load_time = time.time() - start_model_load_time
        print(f"Total time to load model: {1000*total_model_load_time:.1f}ms")

        homtrans = HomTransform(dir)
        # cap=cv2.VideoCapture(0)
        """ for higher resolution (max available: 1920x1080) """
        cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_SETTINGS, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        """ Calibration """
        STransG = homtrans.calibrate(model, cap, sfm=True)

        print("============================")
        print(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

        homtrans.RunGazeOnScreen(model, cap, sfm=True)

        # gocv.PlotPupils(gray_image, prediction, morphedMask, falseColor, centroid)

    except Exception as e:
        print(f"Something wrong when running EyeModel: {e}")

if __name__ == '__main__':
    dir = "C:\\temp\\WebCamGazeEstimation\\"
    main(dir)

