import pathlib
import cv2
import numpy as np

from omegaconf import DictConfig, OmegaConf
from gaze_tracking.homtransform import HomTransform
from plgaze.model_pl_gaze import GazeModel

def main():
    package_root = pathlib.Path(__file__).parent.resolve()
    path = package_root / 'plgaze/data/configs/eth-xgaze.yaml'
    config = OmegaConf.load(path)
    config.PACKAGE_ROOT = package_root.as_posix()
    model = GazeModel(config)

    dir = "."
    homtrans = HomTransform(dir)

    cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_SETTINGS, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    # model.run(cap)
    """ Calibration """
    STransG = homtrans.calibrate(model, cap, sfm=False)

    print("============================")
    print(f"STransG\n{np.array2string(STransG, formatter={'float': lambda x: f'{x:.2f}'})}")

    homtrans.RunGazeOnScreen(model, cap, sfm=False)


if __name__ == '__main__':
    main()