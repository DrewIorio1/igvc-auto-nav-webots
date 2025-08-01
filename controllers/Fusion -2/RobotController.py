"""
    File: RobotController.py - Handles both coordnate and gps for speed navigaton
    
    Required Libraries:
        math - For mathematical operations
        time - For time-related functions
        rclpy - ROS 2 Python client library
        Node - Base class for ROS 2 nodes
        Float32 - Standard message type for floating point numbers
        Twist - Standard message type for velocity commands
    Classes:
        GPSLocalizer - Converts GPS coordinates to local XY coordinates.
            Functions:
            __init__ - Initializes with a reference latitude and longitude.
                to_local_xy - Converts GPS coordinates to local XY coordinates.
        WebotsRosController - Main controller node for the robot, handling navigation and motor control.
            Functions:
            __init__ - Initializes the controller with motors, sensors, and parameters.
            _get_pose - Retrieves the robot's current pose using GPS or a custom localizer.
            _clamp_velocity - Ensures motor velocities stay within limits.
            compute_turn_scale - Computes a scaling factor for turning commands.
            path_set - Sets the path for the robot to follow.
            _follow_path - Calculates linear and angular velocities to follow the path.
            update - Main update loop that computes and sets motor velocities based on the current pose and path.
"""
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from controller import Robot

class GPSLocalizer:
    """
        GPSLocalizer converts GPS coordinates to local XY coordinates based on a reference point.

        It uses the WGS-84 ellipsoid model to calculate distances.

        Variables:
            origin_lat (float): Latitude of the reference point.
            origin_lon (float): Longitude of the reference point.
    """
    def __init__(self, origin_lat, origin_lon):
        """
             Initializes the GPSLocalizer with a reference latitude and longitude.

            Args:
                origin_lat (float): Latitude of the reference point.
                origin_lon (float): Longitude of the reference point.
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

    def to_local_xy(self, lat, lon):
        dx = (lon - self.origin_lon) * 111320 * math.cos(math.radians(self.origin_lat))
        dy = (lat - self.origin_lat) * 111320
        return dx, dy

class WebotsRosController(Node):
    """
        WebotsRosController is a ROS 2 node that controls a robot's motors and navigation.

        It can use GPS for localization or a custom localizer for image-based navigation.

        Variables:
            left_motor (Motor): Left motor of the robot.
            right_motor (Motor): Right motor of the robot.
            use_gps (bool): Flag to indicate if GPS is used for localization.
            gps (GPS): GPS sensor for localization.
            imu (IMU): IMU sensor for orientation data.
            gps_localizer (GPSLocalizer): Converts GPS coordinates to local XY coordinates.
            localize (Localizer): Custom localizer for image-based navigation.
            img_h (int): Height of the image for localization.
            img_w (int): Width of the image for localization.
    """
    def __init__(self, left_motor, right_motor, use_gps=False, gps=None, imu=None,
                 origin_lat=None, origin_lon=None, localize=None, img_h=None, img_w=None):
        super().__init__('robot_controller')
        """
            Initializes the WebotsRosController node.

            Args:
                left_motor (Motor): Left motor of the robot.
                right_motor (Motor): Right motor of the robot.
                use_gps (bool): Flag to indicate if GPS is used for localization.
                gps (GPS): GPS sensor for localization.
                imu (IMU): IMU sensor for orientation data.
                origin_lat (float): Latitude of the reference point for GPS localization.
                origin_lon (float): Longitude of the reference point for GPS localization.
                localize (Localizer): Custom localizer for image-based navigation.
                img_h (int): Height of the image for localization.
                img_w (int): Width of the image for localization.
        """
        # Motor setup with velocity limits
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.max_motor_velocity = 12.3  # Pioneer 3-DX motor limit
        
        self.use_gps = use_gps
        if self.use_gps:
            self.gps = gps
            self.imu = imu
            self.gps_localizer = GPSLocalizer(origin_lat, origin_lon)
            timestep = int(Robot().getBasicTimeStep())
            self.gps.enable(timestep)
            self.imu.enable(timestep)
        else:
            self.localize = localize
            self.img_h = img_h
            self.img_w = img_w

        # Navigation parameters
        self.teleop_v = 0.0
        self.teleop_w = 0.0
        self.last_cmd_time = 0.0
        self.path = []
        self.wp_idx = 0
        self.waypoint_tolerance = 0.1
        self.turn_threshold = 1.0
        self.min_inner_scale = 0.5

        # Safe velocity parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_speed_mph', 1.3),
                ('max_speed_mph', 5.0),
                ('speed_gain', 0.5),  # Reduced gain for safety
                ('wheel_radius', 0.032),
                ('wheel_base', 0.16),
            ]
        )

        # Publishers
        self.left_pub = self.create_publisher(Float32, '/left_wheel_speed', 10)
        self.right_pub = self.create_publisher(Float32, '/right_wheel_speed', 10)

        # Initialize motors with safe velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        #self.create_timer(1/20.0, self.update)

    def _get_pose(self):
        if self.use_gps:
            gps_vals = self.gps.getValues()
            lat = gps_vals[0]
            lon = gps_vals[1]
            x, y = self.gps_localizer.to_local_xy(lat, lon)
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            theta = yaw
        else:
            try:
                self.localize.update()
                x_px, y_px, theta = self.localize.get_pose()
                x = (x_px - self.img_w/2) * 0.01
                y = (y_px - self.img_h/2) * 0.01
            except Exception:
                x, y, theta = 0.0, 0.0, 0.0
        return x, y, theta

    def _clamp_velocity(self, velocity):
        """Ensure velocity stays within motor limits"""
        return max(-self.max_motor_velocity, min(self.max_motor_velocity, velocity))

    def compute_turn_scale(self, w_cmd):
        """
            Computes a scaling factor for turning commands based on the command angular velocity.
            This helps to adjust the speed of the robot during turns, ensuring smoother navigation.

            Args:
            w_cmd (float): Commanded angular velocity in radians per second.
            Returns:
            float: Scaling factor for the turn, ranging from 0 to 1.
        """
        abs_w = abs(w_cmd)
        if abs_w < self.turn_threshold:
            return 1 - (1 - self.min_inner_scale) * (abs_w / self.turn_threshold) ** 2
        return self.min_inner_scale

    def path_set(self, path):
        """
            Sets the path for the robot to follow.

            Args:
                path (list): List of waypoints (x, y) to follow.


        """
        self.path = path
        self.wp_idx = 0

    def _follow_path(self, x, y, theta):
        """
            Follows the path by calculating linear and angular velocities based on the current position

            and orientation of the robot.

            Args:
                x (float): Current x-coordinate of the robot.
                y (float): Current y-coordinate of the robot.
                theta (float): Current orientation of the robot in radians.
            Returns:
                v (float): Linear velocity command in meters per second.
                w (float): Angular velocity command in radians per second.

        """
        k_lin, k_ang = 0.8, 2.0
        v_min = 1.3 * 0.44704
        v_max = 5.0 * 0.44704
        w_max = 1.5
        tol = self.waypoint_tolerance

        if not self.path or self.wp_idx >= len(self.path):
            return 0.0, 0.0

        wx, wy = self.path[self.wp_idx]
        dx, dy = wx - x, wy - y
        dist = math.hypot(dx, dy)

        if dist < tol:
            self.wp_idx = min(self.wp_idx+1, len(self.path)-1)
            wx, wy = self.path[self.wp_idx]
            dx, dy = wx - x, wy - y
            dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        err_yaw = ((target_yaw - theta + math.pi) % (2*math.pi)) - math.pi

        v = max(v_min, min(v_max, k_lin * dist))
        w = max(-w_max, min(w_max, k_ang * err_yaw))
        return v, w

    def update(self):
        """
            Updates the robot's motors based on the current pose and path.
            It calculates the commanded velocities and applies adaptive speed control based on the path
            curvature and distance to the next waypoint.

            Args:
                None
            Returns:
                None



        """
        x, y, theta = self._get_pose()
        now = time.time()

        if now - self.last_cmd_time < 1.0:
            v_cmd, w_cmd = self.teleop_v, self.teleop_w
        else:
            v_cmd, w_cmd = self._follow_path(x, y, theta)

        ps = self.get_parameters([
            'min_speed_mph', 'max_speed_mph',
            'speed_gain', 'wheel_radius', 'wheel_base'
        ])
        min_mph, max_mph, gain, wheel_radius, wheel_base = [p.value for p in ps]

        # Calculate wheel velocities
        left_rad = (v_cmd - 0.5 * w_cmd * wheel_base) / wheel_radius
        right_rad = (v_cmd + 0.5 * w_cmd * wheel_base) / wheel_radius

        # Enhanced adaptive speed control
        path_angle = math.atan2(self.path[self.wp_idx][1] - y, 
                              self.path[self.wp_idx][0] - x)
        angle_diff = abs(((path_angle - theta + math.pi) % (2*math.pi)) - math.pi)
        distance = math.hypot(self.path[self.wp_idx][0]-x, self.path[self.wp_idx][1]-y)
    
        # Dynamic speed adjustment with three regimes
        if angle_diff < 0.3:  # ~17 degrees (straight)
            speed_scale = min(1.0, 0.5 + distance/10.0)  # Ramp up based on distance
        elif angle_diff < 1.0:  # ~57 degrees (moderate turn)
            speed_scale = 0.6
        else:  # Sharp turn
            speed_scale = max(0.3, 0.6 - angle_diff/3.0)

        # Apply boost when aligned and far from goal
        if angle_diff < 0.2 and distance > 5.0:
            speed_scale = min(1.2, speed_scale * 1.3)  # Temporary 30% boost

        left_rad *= speed_scale
        right_rad *= speed_scale

        # Convert to MPH for display
        conversion_factor = wheel_radius * 2.23694
        left_mph = left_rad * conversion_factor
        right_mph = right_rad * conversion_factor

        # Debug prints
        print(f"\nCurrent Goal: {self.path[self.wp_idx]}")
        print(f"Distance: {distance:.2f}m | Heading error: {math.degrees(angle_diff):.1f}°")
        print(f"Speed scaling: {speed_scale:.2f}x")
        print(f"[{time.time():.2f}] Motors: {left_mph:.2f}/{right_mph:.2f} mph")

        # Final safety clamp
        left_rad = self._clamp_velocity(left_rad)
        right_rad = self._clamp_velocity(right_rad)

        # Set motors
        self.left_motor.setVelocity(left_rad)
        self.right_motor.setVelocity(right_rad)
        self.left_pub.publish(Float32(data=left_rad))
        self.right_pub.publish(Float32(data=right_rad))
