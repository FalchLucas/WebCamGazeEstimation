import numpy as np

class Kalman():
    def __init__(self, F, H, Q, R, x_0, P_0) -> None:
        self.F = F
        self.H = H
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        self.x = x_0
        self.P = P_0
        """ delta_t """
        self.dt = 0
        self.dt2 = 0
        self.dt3 = 0
        self.dt4 = 0

    def Predict(self):
        try:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
        except Exception as e:
            print(f"P: \n{self.P}")
            print(f"R: \n{self.R}")
            print(f"Q: \n{self.Q}")
            print(f"Exception in Predict step: {e}")

    def Update(self, z):
        try:
            # s = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            # self.K = self.P @ self.H.T @ s
            self.K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
            self.x = self.x + self.K @ (z - self.H @ self.x)
            self.P = self.P - self.K @ self.H @ self.P
        except Exception as e:
            print(f"P: \n{self.P}")
            print(f"R: \n{self.R}")
            print(f"Q: \n{self.Q}")
            print(f"Exception in Update step: {e}")
    
    def Update_F(self, dt):
        self.dt = dt
        self.dt2 = dt**2
        self.dt3 = dt**3
        self.dt4 = dt**4
        # self.F = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
        self.F = np.eye(2)

    def Update_Q(self, VarQ):
        sigma_x = VarQ[0]
        sigma_y = VarQ[1]
        # sigma_xdot = VarQ[2]
        # sigma_ydot = VarQ[3]
        # self.Q = np.array([[0.25*self.dt4*sigma_xdot + self.dt2*sigma_x, 0.0, 0.5*self.dt3*sigma_xdot, 0.0],
        #             [0.0, 0.25*self.dt4*sigma_ydot + self.dt2*sigma_y, 0.0, 0.5*self.dt3*sigma_ydot],
        #             [0.5*self.dt3*sigma_xdot, 0.0, self.dt2*sigma_xdot, 0.0],
        #             [0.0, 0.5*self.dt3*sigma_ydot, 0.0, self.dt2*sigma_ydot]])
        self.Q = np.array([[0.25*self.dt4*sigma_x, 0.5*self.dt3*sigma_x],[0.5*self.dt3*sigma_y, self.dt2*sigma_y]])

    def Update_R(self, VarR, sig_x):
        VarR[0] += sig_x
        self.R = np.eye(4) * VarR
        
    def get_state(self):
        return self.x, self.P

