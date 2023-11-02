"""
This script is used to compare the gaze estimation results of the OpenVINO model with the Tobii Eye Tracker 5 and Gazerecorder.
The data are stored in sepearte csv files. This script reads the csv files and plots the gaze estimation results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime


# dir = "C:\\tmp\\Trial\\Trial_13Okt23\\"
# dir = "C:\\tmp\\Trial\\Trial_19Okt23\\HeadMovements\\Moving\\"
# dir = "C:\\tmp\\Trial\\Trial_19Okt23\\HeadMovements\\Turning\\"
# dir = "C:\\tmp\\Trial\\Trial_19Okt23\\TrajectoryNoHeadMovements\\"
dir = "C:\\tmp\\Trial\\Trial_19Okt23\\TrajectoryHeadMovements\\"
plt.close('all')


def Mergedf(df1, df2):
    df1 = df1[['timestamp(hh:m:s.ms)','Bgaze_x','Bgaze_y']]
    df2 = df2[['timestamp(hh:m:s.ms)','set_x','set_y','Bgaze_x','Bgaze_y']]
    # Convert the timestamp column to datetime format
    df1['timestamp(hh:m:s.ms)'] = pd.to_datetime(df1['timestamp(hh:m:s.ms)'], format='%H:%M:%S.%f')    
    df2['timestamp(hh:m:s.ms)'] = pd.to_datetime(df2['timestamp(hh:m:s.ms)'], format='%H:%M:%S.%f')

    # Set the timestamp column as the index for both dataframes
    df1 = df1.set_index('timestamp(hh:m:s.ms)')
    df2 = df2.set_index('timestamp(hh:m:s.ms)')

    # Resample both dataframes to a common sampling rate
    df1_resmp = df1.resample('50ms').mean()
    df2_resmp = df2.resample('50ms').mean()

    merged_df = pd.merge(df1_resmp, df2_resmp, left_index=True, right_index=True, how='outer', suffixes=('_L', '_T'))  
    # merged_df.to_csv(os.path.join(dir, "Mergerd.csv"))

    return merged_df

def CompareMyandGazeRecorder(dir):
    dfstats = pd.read_csv(dir+"stats.txt", delimiter='\t')
    width = float(dfstats.loc[26].str.split()[0][1].replace("px,",''))
    height = float(dfstats.loc[27].str.split()[0][1].replace("px,",''))
    width_mm = float(dfstats.loc[26].str.split()[0][2].replace("mm",''))
    height_mm = float(dfstats.loc[27].str.split()[0][2].replace("mm",''))
    dfMY = pd.read_csv(dir+"MyGazeTracking.csv")
    dfTobMy = pd.read_csv(dir+"TobiiGazeTrackingMy.csv")

    dfMY = dfMY[['timestamp(hh:m:s.ms)','Sgaze_x','Sgaze_y']] 
    dfTobMy = dfTobMy[['timestamp(hh:m:s.ms)','set_x','set_y','Sgaze_x','Sgaze_y']]
    dfTobMy = dfTobMy[dfTobMy['Sgaze_x']>0]
    # Convert the timestamp column to datetime format
    dfMY['timestamp(hh:m:s.ms)'] = pd.to_datetime(dfMY['timestamp(hh:m:s.ms)'], format='%Y-%m-%d %H:%M:%S.%f') 
    dfTobMy['timestamp(hh:m:s.ms)'] = pd.to_datetime(dfTobMy['timestamp(hh:m:s.ms)'], format='%Y-%m-%d %H:%M:%S.%f')

    # Set the timestamp column as the index for both dataframes
    dfMY = dfMY.set_index('timestamp(hh:m:s.ms)') 
    dfTobMy = dfTobMy.set_index('timestamp(hh:m:s.ms)')

    # Resample both dataframes to a common sampling rate
    dfMy_resmp = dfMY.resample('50ms').mean()
    dfTobMy_resmp = dfTobMy.resample('50ms').mean()

    df = pd.merge(dfMy_resmp, dfTobMy_resmp, left_index=True, right_index=True, how='outer', suffixes=('_My', '_Tob'))
    df.dropna(inplace=True)

    mse = ((df['Sgaze_x_My'] - df['set_x']) ** 2 + (df['Sgaze_y_My'] - df['set_y']) ** 2).mean()
    mse_mm = ((df['Sgaze_x_My']/width*width_mm - df['set_x']/width*width_mm) ** 2 + (df['Sgaze_y_My']/height*height_mm - df['set_y']/height*height_mm) ** 2).mean()
    rmse = np.sqrt(mse)
    rmse_mm = np.sqrt(mse_mm)
    print(f"RMSE My GazeTracking: {rmse}px -> {rmse_mm}mm")
    mse = ((df['Sgaze_x_Tob'] - df['set_x']) ** 2 + (df['Sgaze_y_Tob'] - df['set_y']) ** 2).mean()
    mse_mm = ((df['Sgaze_x_Tob']/width*width_mm - df['set_x']/width*width_mm) ** 2 + (df['Sgaze_y_Tob']/height*height_mm - df['set_y']/height*height_mm) ** 2).mean()
    rmse = np.sqrt(mse)
    rmse_mm = np.sqrt(mse_mm)
    print(f"RMSE Tobii GazeTracking: {rmse}px -> {rmse_mm}mm")

    """ Compare with Gaze Recorder"""
    dfTobGR = pd.read_csv(dir+"TobiiGazeTrackingGR.csv")
    dfGR = pd.read_csv(dir+"WebcamGazeData.csv")

    dfGR = dfGR[['Timestamp','GazeX','GazeY']]
    dfTobGR = dfTobGR[['timestamp(hh:m:s.ms)','set_x','set_y','Sgaze_x','Sgaze_y']]
    dfTobGR = dfTobGR[dfTobGR['Sgaze_x']>0]
    # Convert the timestamp column to datetime format
    dfGR['Timestamp'] = pd.to_datetime(dfGR['Timestamp'], format='%Y/%m/%d/%H:%M:%S.%f') 
    dfTobGR['timestamp(hh:m:s.ms)'] = pd.to_datetime(dfTobGR['timestamp(hh:m:s.ms)'], format='%Y-%m-%d %H:%M:%S.%f')

    # Set the timestamp column as the index for both dataframes
    dfGR = dfGR.set_index('Timestamp')
    dfTobGR = dfTobGR.set_index('timestamp(hh:m:s.ms)')

    # Resample both dataframes to a common sampling rate
    dfGR_resmp = dfGR.resample('50ms').mean()
    dfTobGR_resmp = dfTobGR.resample('50ms').mean()

    df = pd.merge(dfGR_resmp, dfTobGR_resmp, left_index=True, right_index=True, how='outer', suffixes=('_GR', '_Tob'))
    df.dropna(inplace=True)

    mse = ((df['GazeX'] - df['set_x']) ** 2 + (df['GazeY'] - df['set_y']) ** 2).mean()
    mse_mm = ((df['GazeX']/width*width_mm - df['set_x']/width*width_mm) ** 2 + (df['GazeY']/height*height_mm - df['set_y']/height*height_mm) ** 2).mean()
    rmse = np.sqrt(mse)
    rmse_mm = np.sqrt(mse_mm)
    print(f"RMSE GazeRecorder GazeTracking: {rmse}px -> {rmse_mm}mm")
    mse = ((df['Sgaze_x'] - df['set_x']) ** 2 + (df['Sgaze_y'] - df['set_y']) ** 2).mean()
    mse_mm = ((df['Sgaze_x']/width*width_mm - df['set_x']/width*width_mm) ** 2 + (df['Sgaze_y']/height*height_mm - df['set_y']/height*height_mm) ** 2).mean()
    rmse = np.sqrt(mse)
    rmse_mm = np.sqrt(mse_mm)
    print(f"RMSE Tobii GazeTracking: {rmse}px -> {rmse_mm}mm")

    plt.figure(figsize=(10,10))

    plt.scatter(dfTobMy['set_x'], dfTobMy['set_y'])
    plt.scatter(dfTobMy['Sgaze_x'], dfTobMy['Sgaze_y'])
    plt.scatter(dfMY['Sgaze_x'], dfMY['Sgaze_y'])
    # plt.scatter(dfTobGR['set_x'], dfTobGR['set_y'])
    # plt.scatter(dfGR['GazeX'], dfGR['GazeY'])
    plt.legend(['Target Trajectory', 'TobiiGazeTracking', 'Proposed GazeTracking'])
    plt.xlabel("x-direction on screen in mm")
    plt.ylabel("y-direction on screen in mm")
    plt.grid()

    plt.figure(figsize=(10,10))

    plt.scatter(dfTobGR['set_x'], dfTobGR['set_y'])
    plt.scatter(dfTobGR['Sgaze_x'], dfTobGR['Sgaze_y'])
    plt.scatter(dfGR['GazeX'], dfGR['GazeY'])
    plt.legend(['Target Trajectory', 'TobiiGazeTracking', 'GazeRecorder GazeTracking'])
    plt.xlabel("x-direction on screen in mm")
    plt.ylabel("y-direction on screen in mm")
    plt.grid()



CompareMyandGazeRecorder(dir)
plt.show()