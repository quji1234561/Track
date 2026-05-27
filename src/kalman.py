"""2D Kalman filter for motion prediction.

State vector: X = [x, y, vx, vy]^T
State transition: constant velocity model.
Observation: Z = [x, y]^T

This is a self-implemented Kalman filter using NumPy only.
No OpenCV Kalman or other tracking filters are used.
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
