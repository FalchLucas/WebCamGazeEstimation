import unittest
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from gaze_tracking.model import EyeModel
from gaze_tracking.homtransform import HomTransform

class TestHomTransform(unittest.TestCase):
            
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dir = "C:\\tmp\\GazeEstimation\\"
        self.df = pd.read_csv(self.dir+"results\\Calibration.csv")
        # df = pd.read_csv("..\\..\\..results\\Calibration.csv")
    
    def test_remove_outliers(self):
        ht = HomTransform(self.dir)
        ht.df = self.df
        gaze, SetVal, WTransG, g = ht._RemoveOutliers()
        STransW, scaleWtG, STransG = ht._fitSTransG(gaze, SetVal, g)
        Sg, SgCalib = ht._getCalibValuesOnScreen(g, STransG)
        ht._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")

        # self.assertEqual(result, expected_result)
        plt.show()

    def test_fitSTransG(self):
        ht = HomTransform(self.dir)
        ht.df = self.df
        gaze, SetVal, WTransG, g = ht._RemoveOutliers()
        STransW, scaleWtG, STransG = ht._fitSTransG(gaze, SetVal, g)
        print(f"STransG\n{STransG}")

    def test_AllFunctionInCalibration(self):
        ht = HomTransform(self.dir)
        ht.df = self.df
        gaze, SetVal, WTransG, g = ht._RemoveOutliers()
        STransW, scaleWtG, STransG = ht._fitSTransG(gaze, SetVal, WTransG, g)
        Sg, SgCalib = ht._getCalibValuesOnScreen(g, STransG)  
        """ Plot Gaze On Screen"""
        ht._PlotGaze2D(g, Sg, SgCalib, name="GazeOnScreen")


    def test_RunGazeOnScreen(self):
        video_path = os.path.join(self.dir, video_path)
        cap = cv2.VideoCapture(video_path)
        model = EyeModel(self.dir)
        ht = HomTransform(self.dir)
        ht.RunGazeOnScreen(model, cap)
        

if __name__ == '__main__':
    unittest.main()