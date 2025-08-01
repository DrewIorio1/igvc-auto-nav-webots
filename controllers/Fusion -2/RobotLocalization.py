"""
    File: RobotLocalization.py - A controller for estimating the robot's pose using odometry and optional IMU data.
     
    This module provides functionality to estimate the robot's position and orientation in a 2D plane
    using wheel encoders and, optionally, an Inertial Measurement Unit (IMU). It implements a simple

    complementary filter to fuse the IMU data with the odometry data for improved pose estimation.

    Required Libraries:
    math - For mathematical operations such as trigonometric functions and angle wrapping.

    Classes:
        RobotLocalization - A class that estimates the robot's pose using odometry and optional IMU data.
        
        Functions:
            - __init__: Initializes the localization system with parameters and devices.
            - update: Updates the robot's pose based on encoder readings and optional IMU data.
            - get_pose: Returns the current pose of the robot as a tuple (x, y, heading).

"""
import math

class RobotLocalization:
    """
        A class to estimate the robot's pose using odometry and optional IMU data.

        Variables:
            dt: Time step for the control loop.
            wheel_base: Distance between left and right wheels.
            wheel_radius: Radius of the wheels.
            x: Current x-coordinate of the robot.
            y: Current y-coordinate of the robot.
            heading: Current heading of the robot in radians.

    """
    def __init__(self, time_step, wheel_base_width, wheel_radius, left_motor, right_motor, imu=None):
        """
        time_step         : your control loop interval in seconds
        wheel_base_width  : distance between left and right wheels (meters)
        wheel_radius      : radius of the wheels (meters)
        left_motor        : Webots motor device for the left wheel
        right_motor       : Webots motor device for the right wheel
        imu               : (optional) Webots IMU device
        """
        self.dt = time_step
        self.wheel_base = wheel_base_width
        self.wheel_radius = wheel_radius

        # pose in the world frame
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0   # radians, in [–pi, +pi]

        # keep last encoder readings
        self.last_left_position = 0.0
        self.last_right_position = 0.0

        # store devices
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.imu = imu

        # enable the built-in position sensors
        self.left_encoder = left_motor.getPositionSensor()
        self.right_encoder = right_motor.getPositionSensor()
        self.left_encoder.enable(int(self.dt * 1000))
        self.right_encoder.enable(int(self.dt * 1000))
        if imu:
            imu.enable(int(self.dt * 1000))

    def update(self):
        """
            Update the robot's pose based on encoder readings and optional IMU data.

            This method computes the robot's new position and orientation using
            the wheel encoders and, if available, the IMU. It uses a simple
            complementary filter to fuse the IMU data with the odometry data.

            Returns:
                None
        """
        # 1) read encoders
        left_position = self.left_encoder.getValue()
        right_position = self.right_encoder.getValue()

        # 2) compute how much each wheel turned
        delta_left = left_position - self.last_left_position
        delta_right = right_position - self.last_right_position

        self.last_left_position = left_position
        self.last_right_position = right_position

        # 3) convert to linear distances
        distance_left = delta_left * self.wheel_radius
        distance_right = delta_right * self.wheel_radius

        # 4) average forward distance and heading change
        forward_distance = (distance_left + distance_right) / 2.0
        change_in_heading = (distance_right - distance_left) / self.wheel_base

        # 5) update x,y using the old heading for projection
        old_heading = self.heading
        self.x += forward_distance * math.cos(old_heading)
        self.y += forward_distance * math.sin(old_heading)

        # 6) update heading and wrap to [-pi, +pi]
        self.heading += change_in_heading
        self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi

        # 7) optional IMU fusion (simple complementary filter)
        if self.imu:
            imu_roll, imu_pitch, imu_yaw = self.imu.getRollPitchYaw()
            # assume imu_yaw is in radians and same frame as our heading
            alpha = 0.98  # tune between 0 (all IMU) and 1 (all odometry)
            self.heading = alpha * self.heading + (1 - alpha) * imu_yaw
            # re‐wrap after fusion
            self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi

    def get_pose(self):
        """
            get the current pose of the robot

            Returns:
                A tuple (x, y, heading) representing the robot's position and orientation.
        """
        return self.x, self.y, self.heading

