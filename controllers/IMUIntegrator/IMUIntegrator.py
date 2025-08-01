#!/usr/bin/env python3
"""
    IMU Integrator Node for ROS2 using Kalman Filter for 1D position and velocity estimation.

    This node subscribes to IMU data and integrates the linear acceleration to estimate the position
    and velocity of the robot. It can use a Kalman filter for better noise handling or direct integration.

    This node is designed to work with the intel camera's imu

    Required Libraries:
        rclpy - ROS2 Python client library
        rclpy.node - Base class for ROS2 nodes
        sensor_msgs.msg - ROS2 message types for sensor data
        geometry_msgs.msg - ROS2 message types for geometry
        numpy - Library for numerical operations
        math - Library for mathematical functions for trigonometric calculations of sine and cosine

    Publishes:
        Pose Stamped - The estimated pose of the robot, including position and orientation.

    Subscribes:
        IMU - IMU data containing linear acceleration and angular velocity.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
from math import sin, cos

class KalmanFilter1D:
    """
    A simple 1D Kalman filter for estimating position (x) and velocity (v_x).

    State vector: [position, velocity]^T

    Variables:
        dt: Time step for the filter.
        x: State vector (position and velocity).
        P: Error covariance matrix.
        F: State transition matrix.
        B: Control input matrix.
        Q: Process noise covariance.
        H: Measurement matrix.
        R: Measurement noise covariance.

    """
    def __init__(self, dt, measurement_noise, process_noise):
        """
            Constructor for the KalmanFilter1D class.

            Args:
                dt: Time step for the filter.
                measurement_noise: Measurement noise covariance.
                process_noise: Process noise covariance.
        """
        self.dt = dt
        # Initial state: zero position and velocity.
        self.x = np.array([[0.0], [0.0]])
        # Initial error covariance.
        self.P = np.eye(2)
        # State transition matrix (will update with dt).
        self.F = np.array([[1, dt],
                           [0, 1]])
        # Control input matrix.
        self.B = np.array([[0.5 * dt ** 2],
                           [dt]])
        # Process noise covariance.
        self.Q = process_noise * np.eye(2)
        # Measurement matrix: here, we assume we can observe the position.
        self.H = np.array([[1, 0]])
        # Measurement noise covariance.
        self.R = np.array([[measurement_noise]])

    def predict(self, u, dt):
        """
            Predict the next state and update the error covariance.

            Args:
                u: Control input (acceleration).
                dt: Time step for the prediction.

            Returns:
                x: Predicted state vector (position and velocity).
        """

        # Update dt-dependent matrices.
        self.F = np.array([[1, dt],
                           [0, 1]])
        self.B = np.array([[0.5 * dt ** 2],
                           [dt]])
        # Predict the state: x = F * x + B * u
        self.x = np.dot(self.F, self.x) + self.B * u
        # Predict error covariance.
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        """
            Update the state with a new measurement.

            Args:
                z: Measurement (position).

            Returns:
                x: Updated state vector (position and velocity).
        """
        y = z - np.dot(self.H, self.x)
        # Innovation covariance.
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Kalman gain.
        K = np.dot(self.P, self.H.T) @ np.linalg.inv(S)
        # Update state estimate.
        self.x = self.x + np.dot(K, y)
        # Update error covariance.
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

class IMUIntegrator(Node):
    """
        IMU Integrator Node for ROS2.

        This node subscribes to IMU data and integrates the linear acceleration
        to estimate the position and velocity of the robot. It can use a Kalman

        Variables:
            use_kf: Boolean to choose between Kalman filter and direct integration.
            last_msg_time: Timestamp of the last received IMU message.
            theta: Orientation (yaw in radians).
            x_position: Estimated x position.
            velocity_x: Estimated x velocity.

    """

    def __init__(self):
        """
            Constructor for the IMUIntegrator class.

            Initializes the node, subscribes to IMU data, and sets up the publisher.


        """
        super().__init__('imu_integrator')
        
        # Declare a parameter to choose the integration method.
        self.declare_parameter('use_kalman', True)
        self.use_kf = self.get_parameter('use_kalman').value
        self.get_logger().info("Using Kalman Filter: " + str(self.use_kf))
        
        # Subscribe to the IMU topic (ensure the topic name matches your setup).
        self.subscription = self.create_subscription(
            Imu,
            'IMU',  # May need to change...
            self.imu_callback,
            10
        )
        
        # Publisher for the updated pose.
        self.pose_publisher = self.create_publisher(PoseStamped, 'pose', 10)
        
        # Variables for tempo-spatial integration.
        self.last_msg_time = None
        self.theta = 0.0  # Orientation (yaw in radians).
        
        # Initialization for x position integration.
        if self.use_kf:
            # Initialize the Kalman filter with nominal dt and noise parameters.
            self.kf = KalmanFilter1D(dt=0.1, measurement_noise=0.1, process_noise=0.01)
        else:
            self.x_position = 0.0  # Direct integration of position.
            self.velocity_x = 0.0  # Direct integration of velocity.

    def imu_callback(self, msg: Imu):
        """
            Callback function for IMU data.

            Args:
                msg: The IMU message containing linear acceleration and angular velocity.
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_msg_time is None:
            self.last_msg_time = current_time
            return
        dt = current_time - self.last_msg_time
        self.last_msg_time = current_time

        # --- Orientation Integration ---
        # Update orientation (theta) from the angular velocity (assumed around the z-axis).
        omega_z = msg.angular_velocity.z
        self.theta += omega_z * dt

        # --- x Position Integration ---
        acceleration_x = msg.linear_acceleration.x
        if self.use_kf:
            # Use the Kalman filter to integrate x acceleration.
            state_pred = self.kf.predict(u=acceleration_x, dt=dt)
            # For demonstration, simulate a measurement by using the prediction.
            # In a real system, replace 'z' with an actual measurement.
            z = np.array([[state_pred[0, 0]]])
            state_est = self.kf.update(z)
            filtered_x = state_est[0, 0]
        else:
            # Direct integration: first update velocity then position.
            self.velocity_x += acceleration_x * dt
            self.x_position += self.velocity_x * dt
            filtered_x = self.x_position

        # --- Prepare the PoseStamped Message ---
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        
        # Assign the computed x position.
        pose_msg.pose.position.x = filtered_x
        pose_msg.pose.position.y = 0.0  # Assuming movement is along x.
        pose_msg.pose.position.z = 0.0
        
        # Convert the yaw (theta) to a quaternion representation.
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = sin(self.theta / 2.0)
        pose_msg.pose.orientation.w = cos(self.theta / 2.0)

        # Publish the pose.
        self.pose_publisher.publish(pose_msg)
        self.get_logger().info(
            f"dt: {dt:.4f}s, θ: {self.theta:.4f} rad, x: {filtered_x:.4f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = IMUIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("IMUIntegrator node terminated manually")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()