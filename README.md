# Webcam based Gaze Estimation
Estimate gaze on computer screen

## Installation Guidelines

- [ ] Set Up Python Virtual Environment

    Create a virtual Python environment to avoid dependency conflicts. To create a virtual environment (Python 3.8), use the following command or create a virtual environment with Anaconda navigator:

        conda create --name <env_name> python=3.8

- [ ] Activate Virtual Environment

    Activate the newly created Python virtual environment by issuing this command:    

        conda activate <env_name>

- [ ] Set Up and Update PIP to the Highest Version

    Make sure pip is installed in your environment and upgrade it to the latest version by issuing the following command:

        python -m pip install --upgrade pip

- [ ] Install the Package

    To install OpenVINO Development Tools into the existing environment with the deep learning framework of your choice, run the following command:

        pip install openvino-dev
        pip install -r requirements.txt or conda install --file requirements.txt

- [ ] Copy the file openh264-1.8.0-win64.dll into the environment path (e.g. C:\Anaconda3\envs\name_of_env)

- [ ] run main.py to test the application

- [ ] in order to run main_compareWithTobii.py you need to generate a exe file that runs Tobii Eye Tracker 5, for this you need the sdk dll for Tobii Eye Tracker 5