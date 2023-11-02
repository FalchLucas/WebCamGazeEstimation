import cv2
import os
import time
import numpy as np
import screeninfo
import matplotlib.pyplot as plt


class Targets():
    def __init__(self, width, height):
        self.move = 1
        
        self.width = width
        self.height = height
        self.SetPos = [int(self.width/2), int(self.height/2)]

        self.tstart = 0
        self.x1 = 0
        self.x2 = self.width
        self.x3 = 0
        self.x4 = self.width
        self.y5 = 0
        self.timerandom_start = 0
        self.randpos1 = False   
        self.randpos2 = False
        self.randpos3 = False
        self.randpos4 = False
        self.ch_x = True

    def getTargetCalibration(self, time_interval=2):
        """
        This function is used to get the target positions for calibration.
        By uncommenting the code, you can get more target positions.
        """
        tstop = time.time()
        tdelta = tstop-self.tstart
        idx = 4
        if tdelta < time_interval:
            idx = 0
            Setpos = [int(self.width/10), int(self.height/10)]
        elif tdelta > time_interval and tdelta < 2*time_interval:        
            idx = 1
            Setpos = [int(self.width*9/10), int(self.height/10)]
        elif tdelta > 2*time_interval and tdelta < 3*time_interval:
            idx = 2
            Setpos = [int(self.width/10), int(self.height*9/10)]
        elif tdelta > 3*time_interval and tdelta < 4*time_interval:
            idx = 3
            Setpos = [int(self.width*9/10), int(self.height*9/10)]
        # elif tdelta > 4*time_interval and tdelta < 5*time_interval:
        #     idx = 4
        #     Setpos = [int(self.width/2), int(self.height/2)]
        # elif tdelta > 5*time_interval and tdelta < 6*time_interval:
        #     idx = 5
        #     Setpos = [int(self.width/2), int(self.height/8)]
        # elif tdelta > 6*time_interval and tdelta < 7*time_interval:
        #     idx = 6
        #     Setpos = [int(self.width/2), int(self.height*9/10)]
        # elif tdelta > 7*time_interval and tdelta < 8*time_interval:
        #     idx = 7
        #     Setpos = [int(self.width/10), int(self.height/2)]
        # elif tdelta > 8*time_interval and tdelta < 9*time_interval:
        #     idx = 8
        #     Setpos = [int(self.width*9/10), int(self.height/2)]
        # elif tdelta > 9*time_interval and tdelta < 10.5*time_interval:
        #     idx = 9
        #     Setpos = [int(self.width/2), int(self.height/2)]
        # elif tdelta > 10.5*time_interval and tdelta < 12*time_interval:
        #     idx = 10
        #     Setpos = [int(self.width/2), int(self.height/2)]
        # elif tdelta > 12*time_interval and tdelta < 13.5*time_interval:
        #     idx = 11
        #     Setpos = [int(self.width/2), int(self.height/2)]
        # elif tdelta > 13.5*time_interval and tdelta < 15*time_interval:
        #     idx = 12
        #     Setpos = [int(self.width/2), int(self.height/2)]

        else:
            print(f"Calibration Done.")
            idx = None
            Setpos = np.array([0, 0])
        
        return idx, Setpos

    def getTargetOnScreen(self, time_interval=2):
        
        tstop = time.time()
        tdelta = tstop-self.tstart  
        if tdelta < time_interval:
            self.SetPos = [int(self.width/8), int(self.height/8)]
        elif tdelta > time_interval and tdelta < 2*time_interval:        
            self.SetPos = [int(self.width*7/8), int(self.height/8)]
        elif tdelta > 2*time_interval and tdelta < 3*time_interval:
            self.SetPos = [int(self.width/8), int(self.height*7/8)]
        elif tdelta > 3*time_interval and tdelta < 4*time_interval:
            self.SetPos = [int(self.width*7/8), int(self.height*7/8)] 
        elif tdelta > 4*time_interval and tdelta < 5*time_interval:
            self.SetPos = [10, int(self.height/2)]
        elif tdelta > 5*time_interval and tdelta < 6*time_interval:
            self.SetPos = [int(self.width-10), int(self.height/2)]
        elif tdelta > 6*time_interval and tdelta < 7*time_interval:
            self.SetPos = [int(self.width/2), 10]
        elif tdelta > 7*time_interval and tdelta < 8*time_interval:
            self.SetPos = [int(self.width/2), int(self.height-10)]
        elif tdelta > 8*time_interval and self.x1 <= self.width:            
            y = (self.height/self.width)*self.x1
            self.SetPos = [self.x1, int(y)]
            self.x1 += 10
        elif tdelta > 8*time_interval and self.x1 > self.width and self.x2 >= 0:
            y = self.height
            self.SetPos = [self.x2, int(y)]
            self.x2 -=10
        elif tdelta > 8*time_interval and self.x1 > self.width and self.x2 < 0 and self.x3 <= self.width:
            y = -(self.height/self.width)*self.x3 + self.height
            self.SetPos = [self.x3, int(y)]
            self.x3 += 10
        elif tdelta > 8*time_interval and self.x1 > self.width and self.x2 < 0 and self.x3 > self.width and self.x4 >= 0:
            y = 0
            self.SetPos = [self.x4, int(y)]
            self.x4 -= 10
        elif tdelta > 8*time_interval and self.x1 > self.width and self.x2 < 0 and self.x3 > self.width and self.x4 < 0 and self.y5 <= self.height:
            self.SetPos = [0, self.y5]
            self.y5 += 10
        elif tdelta > 8*time_interval and self.x1 > self.width and self.x2 < 0 and self.x3 > self.width and self.x4 < 0 and self.y5 > self.height and self.timerandom_start == 0:
            self.timerandom_start = tstop
            self.SetPos = self._getRandomPos()
        elif self.timerandom_start > 0 and tstop-self.timerandom_start < time_interval:
            pass
        elif self.timerandom_start > time_interval and self.randpos1 == False:
            self.SetPos = self._getRandomPos()
            self.randpos1 = True
        elif self.timerandom_start > time_interval and tstop-self.timerandom_start < 2*time_interval:
            pass
        elif self.timerandom_start > 2*time_interval and self.randpos2 == False:
            self.SetPos = self._getRandomPos()
            self.randpos2 = True
        elif self.timerandom_start > 2*time_interval and tstop-self.timerandom_start < 3*time_interval:                
            pass
        elif self.timerandom_start > 3*time_interval and self.randpos3 == False:
            self.SetPos = self._getRandomPos()
            self.randpos3 = True
        elif self.timerandom_start > 3*time_interval and tstop-self.timerandom_start < 4*time_interval:                
            pass
        elif self.timerandom_start > 4*time_interval and self.randpos4 == False:
            self.SetPos = self._getRandomPos()
            self.randpos4 = True
        elif self.timerandom_start > 4*time_interval and tstop-self.timerandom_start < 5*time_interval:                
            pass
        elif self.randpos1 == True and self.randpos2 == True and self.randpos3 == True and self.randpos4 == True:
            self.tstart = time.time()
            self.x1 = 0
            self.x2 = self.width
            self.x3 = 0
            self.x4 = self.width
            self.y5 = 0
            self.timerandom_start = 0
            self.randpos1 = False   
            self.randpos2 = False
            self.randpos3 = False
            self.randpos4 = False
        else:            
            print("Error")

        return self.SetPos

    def _getRandomPos(self):
        return [np.random.randint(0, self.width), np.random.randint(0, self.height)]

    def setSetPos(self, SetPos):
        self.SetPos = SetPos

    def DrawSingleTargets(self, disp_frame, gaze = np.array([-10,-10]), time_interval=2):
        frame = disp_frame.copy()

        tstop = time.time()
        tdelta = tstop-self.tstart

        if tdelta > time_interval:
            self.SetPos = [int(np.random.rand()*self.width), int(np.random.rand()*self.height)]
            self.tstart = time.time()

        cv2.drawMarker(frame, tuple(self.SetPos), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=4) 

        # cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 15, (0,0,255), -1)   # -1 to fill the circle

        cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze", frame)
        return frame, self.SetPos

    def DrawTargetInMiddle(self, disp_frame, gaze):
        frame = disp_frame.copy()

        self.SetPos[0] = int(self.width/2)
        self.SetPos[1] = int(self.height/2)
        cv2.drawMarker(frame, tuple(self.SetPos), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=4) 

        cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 15, (0,0,255), -1)   # -1 to fill the circle

        cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze", frame)
        return frame, self.SetPos

    def DrawRectangularTargets(self, disp_frame, gaze, step = 20, offset  = 10):
        frame = disp_frame.copy()

        step_x = 0
        step_y = 0

        if self.ch_x == True:
            if self.SetPos[0]+step_x < int(self.width*7/8) and self.move%2 == 1 and self.SetPos[1]+step_y < int(self.height-offset):
                step_x = step
                step_y = 0
            elif self.SetPos[0]+step_x > int(self.width/8) and self.move%2 == 0 and self.SetPos[1]+step_y < int(self.height-offset):
                step_x = -step
                step_y = 0
            elif self.SetPos[1]+self.height/4 < int(self.height-offset):
                self.move += 1
                step_x = 0
                step_y = self.height/4
            else:
                step_x = 0
                step_y = -20
                self.ch_x = False
                self.move = 1
        else:
            if self.SetPos[1] > int(self.height/8) and self.move%2 == 1 :
                step_x = 0
                step_y = -20
            elif self.SetPos[1] < int(self.height*7/8) and self.move%2 == 0:
                step_x = 0
                step_y = 20
            elif self.SetPos[0] < int(self.width*7/8):
                self.move += 1
                step_x = self.width/4
                step_y = 0
            else:
                step_x = 0
                step_y = 0

        self.SetPos[0] = int(self.SetPos[0]+step_x)
        self.SetPos[1] = int(self.SetPos[1]+step_y)
        cv2.drawMarker(frame, tuple(self.SetPos), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=4) 

        # cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 15, (0,0,255), -1)   # -1 to fill the circle

        cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze", frame)
        return frame, self.SetPos

    def DrawTargetGaze(self, disp_frame, gaze):
        frame = disp_frame.copy()

        step_x = 0
        step_y = 0
        if (self.SetPos[0]+10) < self.width and (self.SetPos[1]-10) > 0 and self.move <= 1:
            self.move = 1
            step_x = 10
            step_y = -10
        elif (self.SetPos[1]+10) < self.height and self.move <= 2:
            self.move = 2
            step_x = 1
            step_y = 10
        elif (self.SetPos[0]-10) > 0 and (self.SetPos[1]-10) > 0 and self.move <= 3:
            self.move = 3
            step_x = -10
            step_y = -10
        elif (self.SetPos[1]+10) < self.height and self.move <= 4:
            self.move = 4
            step_x = -5
            step_y = 10
            self.tstart = time.time()
        else:
            self.move = 5        
            step_x = 0
            step_y = 0
            self.getTargetOnScreen()
        
        self.SetPos[0] = int(self.SetPos[0]+step_x)
        self.SetPos[1] = int(self.SetPos[1]+step_y)
        cv2.drawMarker(frame, tuple(self.SetPos), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=4)
        
        if np.size(gaze) <= 3:
            # cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 15, (0,0,255), -1)   # -1 to fill the circle
            pass
        else:
            cx,cy,cz = np.mean(gaze, axis=1)
            cv2.circle(frame, (int(gaze[0][0]), int(gaze[1][0])), 15, (0,0,0), -1)   # -1 to fill the circle (bgr)
            cv2.circle(frame, (int(gaze[0][1]), int(gaze[1][1])), 15, (0,0,0), -1)   # -1 to fill the circle
            cv2.circle(frame, (int(gaze[0][2]), int(gaze[1][2])), 15, (0,0,0), -1)
            cv2.circle(frame, (int(gaze[0][3]), int(gaze[1][3])), 15, (0,0,0), -1)
            cv2.circle(frame, (int(cx), int(cy)), 15, (0,0,255), -1)   # -1 to fill the circle

        cv2.namedWindow("Gaze", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze", frame)
        return frame, self.SetPos

def getVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def getWhiteFrame(height, width):
    return 255*np.ones((width, height, 3), dtype=np.uint8)

def get_out_video(cap, output_path, file_name = "calibrate.mp4", scalewidth = 1):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # get frame width in pixel
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # get frame height in pixel
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 20
    out_video = cv2.VideoWriter( os.path.join(output_path,file_name), cv2.VideoWriter_fourcc(*'avc1'), fps, (scalewidth*width, height))
    return out_video, width, height

def display_window(frame):
    window_name = "CalibWindow"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)    
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, frame)

def getWebcamSize(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # get frame width in pixel
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # get frame height in pixel
    print(f"Webcam Size: {width}x{height}")
    return width, height

def getScreenSize():
    screen = screeninfo.get_monitors()
    for s in screen:
        if s.is_primary:
            width = s.width
            height = s.height
            width_mm = s.width_mm
            height_mm = s.height_mm
    print(f"Screen Size: {width}x{height}")
    return width, height, width_mm, height_mm

def ReadCameraCalibrationData(calibration_path, file_name = "calibration_data.txt"):
    path = os.path.join(calibration_path, file_name)
    with open(path, 'r') as f:
        lines = f.readlines()

    # Extract the camera matrix and distortion coefficients from the file
    camera_matrix = np.array([[float(x) for x in lines[1].split()],
                            [float(x) for x in lines[2].split()],
                            [float(x) for x in lines[3].split()]])

    dist_coeffs = np.array([float(x) for x in lines[5].split()])

    return camera_matrix, dist_coeffs

def PlotPupils(gray_image, prediction, morphedMask, falseColor, centroid):
    THRESHOLD = 0.5
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15,5))
    ax[0].imshow(gray_image, cmap='gray')
    ax[0].set_title('Eye image')

    ax[1].imshow(prediction, vmin=0, vmax=1)
    ax[1].set_title('Probability map')

    ax[2].imshow(prediction > THRESHOLD)
    ax[2].set_title('Map Binarized')

    ax[3].imshow(morphedMask)
    ax[3].set_title('After Morphology')

    ax[4].imshow(falseColor)
    ax[4].plot(centroid[1], centroid[0], marker='+', markersize=12, mew=2, mec='g')
    ax[4].set_title('Binary pupil')
    plt.show()