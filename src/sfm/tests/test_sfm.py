import unittest
import pandas as pd
import matplotlib.pyplot as plt
import os

from sfm.sfm_module import SFM
from gaze_tracking.model import EyeModel


class TestSFM(unittest.TestCase):
            
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dir = "C:\\Users\\lucas.falch\\OneDrive - OST\\Dokumente\\Projects\\Innovation Visualization Tools for Big Battery Data\\Coding\\OpenVINO\\MyOpenVino\\"

    def test_sfm_video(self):
        sfm = SFM(self.dir)
        model = EyeModel(self.dir)
        # video_path = os.path.join(self.dir, "results", "output_video.mp4")
        video_path = os.path.join(self.dir, "results", "calibrate.mp4")
        sfm.sfm_video(model, video_path)



if __name__ == '__main__':
    unittest.main()