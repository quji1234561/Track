"""自写二维卡尔曼滤波器 —— 纯NumPy矩阵运算实现。

状态向量 X = [x, y, vx, vy]^T  (位置+速度)
观测向量 Z = [x, y]^T            (仅观测位置)

恒速模型: 假设目标在两帧之间匀速直线运动。
  x(t+1) = x(t) + vx*dt
  y(t+1) = y(t) + vy*dt

predict(): 用状态转移矩阵A预测下一时刻位置 → (pred_x, pred_y)
update(): 用观测值Z通过卡尔曼增益K修正状态估计 → (filtered_x, filtered_y)

为什么需要卡尔曼: 当NCC短暂失败时，卡尔曼可用速度估计继续预测目标位置，
保持轨迹连续。速度从观测序列中自动学习。

本实现完全不使用cv2.KalmanFilter，所有矩阵运算由NumPy完成。
"""

import numpy as np


class KalmanFilter2D:
    """2D Kalman filter for center-point motion prediction."""

    def __init__(self, x, y, dt=1.0):
        """Initialize filter with initial position.

        Args:
            x: initial x coordinate.
            y: initial y coordinate.
            dt: time step between frames (default 1 frame).
        """
        # State: [x, y, vx, vy]
        self.X = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)

        # State transition matrix
        self.dt = dt
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # Observation matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * 0.01

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 5.0

        # Error covariance
        self.P = np.eye(4, dtype=np.float32) * 10.0

    def predict(self):
        """Predict next state.

        Returns:
            (pred_x, pred_y) predicted center point.
        """
        self.X = self.A @ self.X
        self.P = self.A @ self.P @ self.A.T + self.Q
        return float(self.X[0, 0]), float(self.X[1, 0])

    def update(self, x, y):
        """Update filter with observed measurement.

        Args:
            x: observed x coordinate.
            y: observed y coordinate.

        Returns:
            (updated_x, updated_y) filtered center point.
        """
        Z = np.array([[x], [y]], dtype=np.float32)

        # Innovation / residual
        Y = Z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.X = self.X + K @ Y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        return float(self.X[0, 0]), float(self.X[1, 0])

    @property
    def velocity(self):
        """Current velocity estimate (vx, vy)."""
        return float(self.X[2, 0]), float(self.X[3, 0])
