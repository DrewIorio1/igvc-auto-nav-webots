#!/usr/bin/env python3
"""
    File: RobotController.py - ROS2 node for differential-drive Webots robot



    Required Libraries:

        rclpy - ROS2 Python client library
        rclpy.node - Base class for ROS2 nodes
        rclpy.time - Time utilities for ROS2 to provide time and duration
        rclpy.duration - Duration utilities for ROS2
        tf2_ros - ROS2 library for handling transforms
        geometry_msgs.msg - ROS2 message types for geometry for Twist and Path
        nav_msgs.msg - ROS2 message types for navigation for Path
        numpy - Library for numerical operations
        math - Library for mathematical functions

    Required Resouces:
        params.yaml -  
          - min_speed_mph, max_speed_mph, speed_gain  
          - wheel_radius, wheel_base  

    Ros Publish
        Publishes left/right wheel speeds (rad/s) on Float32 topics  


    Methods:
        __init__(self, left_motor, right_motor, localize, img_h, img_w) :
            Initializes the WebotsRosController class with motors and localization.
        update(self):
            Updates the robot's speed based on localization and path following.
        _follow_path:
            Follows the path using a simple proportional controller.
        path_set:
            Set the path to follow

        _get_pose:
            gets the current pose from the localization
        
            



"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

from controller import Robot  # your Webots Robot
# euler_from_quaternion omitted since we use image-based localization

class WebotsRosController(Node):
    """
        WebotsRosController - Robot Controller for Webots Simulation
        This class implements a ROS2 node that controls a differential drive robot 

        Attributes:
            left_motor: The left motor of the robot.
            right_motor: The right motor of the robot.
            localizatoin: The localization controller for the robot.
            h: The height of the robot in the simulation.
    """

    def __init__(self, left_motor, right_motor, localize, img_h, img_w):
        super().__init__('robot_controller')

        # Webots interfaces
        self.left_motor  = left_motor
        self.right_motor = right_motor
        self.localize   = localize
        self.img_h, self.img_w = img_h, img_w

        # Teleop override
        self.teleop_v = 0.0
        self.teleop_w = 0.0
        self.last_cmd_time = 0.0

        # Path following
        self.path = []
        self.wp_idx = 0
        self.waypoint_tolerance = 0.1  # m
        
        self.turn_threshold = 1.0  # Example threshold for scaling
        self.min_inner_scale = 0.5  # Minimum allowed scaling

        
        # --- Load all parameters at once ---
        self.declare_parameters(
            namespace='',
            parameters=[
                ('min_speed_mph', 1.3),
                ('max_speed_mph', 5.0),
                ('speed_gain',    1.0),
                ('wheel_radius',  0.032),
                ('wheel_base',    0.16),
            ]
        )

        # Subscribers
        #self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)
        #self.create_subscription(Path,  '/path',    self._on_path,    10)

        # Publishers
        self.left_pub  = self.create_publisher(Float32, '/left_wheel_speed',  10)
        self.right_pub = self.create_publisher(Float32, '/right_wheel_speed', 10)

        # Set Webots motor max velocity (rad/s) based on max_speed_mph
        max_mph      = self.get_parameter('max_speed_mph').value
        wheel_radius = self.get_parameter('wheel_radius').value
        max_radps    = (max_mph * 0.44704) / wheel_radius
        self.left_motor .setVelocity(max_radps)
        self.right_motor.setVelocity(max_radps)

        # Run update() at 20 Hz
        #self.create_timer(1/20.0, self.update)

    def _on_cmd_vel(self, msg: Twist):
        """ Teleop override: use /cmd_vel for 1 s """
        self.teleop_v      = msg.linear.x
        self.teleop_w      = msg.angular.z
        self.last_cmd_time = time.time()

    def path_cb(self, msg: Path):
        """ Receive a new Path, reset waypoint index """
        self.path   = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.wp_idx = 0
    

    def path_set(self, path):
        """
            Set the path for the robot to follow.

            Args:
                path: A list of tuples containing the path points (x, y).


        """
        self.path = path
        self.wp_idx = 0

    def _get_pose(self):
        """
            Get the current pose of the robot in the world frame.

            returns: 
                x, y, theta
        """

        """
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', now, timeout=rclpy.duration.Duration(seconds=0.1))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            """
        try:
            self.localize.update()
            x_px, y_px, theta = self.localize.get_pose()
            x = (x_px - self.img_w/2) * 0.01
            y = (y_px - self.img_h/2) * 0.01
            return x, y, theta
        except Exception:
            return 0.0, 0.0, 0.0



    def compute_turn_scale(self, w_cmd):
        """
        Compute a smooth scaling factor for inner wheel speed.

        When |w_cmd| is zero, the scale is 1 (no bias).
        As |w_cmd| increases toward `self.turn_threshold`, the scale decays quadratically toward `self.min_inner_scale`.
        For |w_cmd| >= threshold, the scale remains at `self.min_inner_scale`.

        Parameters:
            w_cmd (float): Angular command.

        Returns:
            float: scaling factor between `self.min_inner_scale` and 1.
        """
        abs_w = abs(w_cmd)
        if abs_w < self.turn_threshold:
            return 1 - (1 - self.min_inner_scale) * (abs_w / self.turn_threshold) ** 2
        else:
            return self.min_inner_scale

    
    def _follow_path(self, x, y, theta):
        """
        Follow the current path using a simple proportional controller.
        Returns (v, w): linear and angular velocity commands.
        """
        k_lin, k_ang = 0.8, 2.0
        v_min = 1.3 * 0.44704
        v_max = 5.0 * 0.44704
        w_max = 1.5
        tol   = self.waypoint_tolerance

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
        err_yaw    = ((target_yaw - theta + math.pi) % (2*math.pi)) - math.pi

        v = max(v_min, min(v_max, k_lin * dist))
        w = max(-w_max, min(w_max, k_ang * err_yaw))
        return v, w
    
    def update(self):
        """
        Update the speed of the motors based on the current pose and path.
        """
        import math, time
        from std_msgs.msg import Float32
    
        # 1) Get current pose and determine v_cmd, w_cmd.
        x, y, theta = self._get_pose()
        now = time.time()
        
        if now - self.last_cmd_time < 1.0:
            v_cmd, w_cmd = self.teleop_v, self.teleop_w
        else:
            v_cmd, w_cmd = self._follow_path(x, y, theta)
    
        # 2) Fetch parameters.
        ps = self.get_parameters([
            'min_speed_mph', 'max_speed_mph',
            'speed_gain', 'wheel_radius', 'wheel_base'
        ])
        min_mph, max_mph, gain, wheel_radius, wheel_base = [p.value for p in ps]
        
        # 3) Compute raw wheel speeds using inverse kinematics (in rad/s).
        left_rad  = (v_cmd - 0.5 * w_cmd * wheel_base) / wheel_radius
        right_rad = (v_cmd + 0.5 * w_cmd * wheel_base) / wheel_radius
    
        # 4) Convert rad/s to mph.
        factor    = wheel_radius * 2.23694
        left_mph  = left_rad * factor
        right_mph = right_rad * factor
    
        # 5) Smooth bias for the inner wheel.
        # Here, we use our compute_turn_scale function.
        # You can choose which wheel is "inner" based on the sign of w_cmd.
        # For w_cmd > 0, assume turning left (slow left wheel).
        # For w_cmd < 0, assume turning right (slow right wheel).
        # Adjust the threshold and min_scale parameters as needed.
        
        # For smoother bias, we combine the gain with the w_cmd before applying the scale.
        # You might experiment with these values.
        turn_threshold = 1.0   # turn rate at which maximum bias is reached
        min_inner_scale = 0.5  # inner wheel speed will be scaled no lower than 50%
        
        if w_cmd > 0:
            # Turning left: slow the left wheel (inner)
            turn_scale = self.compute_turn_scale(w_cmd * gain)
            left_mph *= turn_scale
        elif w_cmd < 0:
            # Turning right: slow the right wheel (inner)
            turn_scale = self.compute_turn_scale(w_cmd * gain)
            right_mph *= turn_scale
        else:
            # No turning: both wheels remain as computed.
            turn_scale = 1.0
        print(f"w_cmd: {w_cmd}, Turn Scale: {turn_scale}")
        
        left_mph = min(max(left_mph, min_mph), max_mph)
        right_mph = min(max(right_mph, min_mph), max_mph)
        # Debug print for monitoring.
        print(f"Left mph {left_mph:.3f} | Right mph {right_mph:.3f} | Turn_scale {turn_scale:.3f} | "
              f"V_cmd {v_cmd:.3f} | W_cmd {w_cmd:.3f} | Wheel_radius {wheel_radius} | "
              f"Wheel_base {wheel_base} | Factor {factor:.5f} | Gain {gain} | "
              f"Min_mph {min_mph} | Max_mph {max_mph}")
    
        # 6) Clamp speeds to [min_mph, max_mph].
        def clamp(v):
            return math.copysign(max(min_mph, min(max_mph, abs(v))), v) if abs(v) > 0 else 0.0
    
        left_mph  = clamp(left_mph)
        right_mph = clamp(right_mph)
    
        # 7) Convert back to rad/s before sending to motors.
        left_rad  = left_mph / factor
        right_rad = right_mph / factor
    
        # 8) Command the motors.
        self.left_motor.setVelocity(left_rad)
        self.right_motor.setVelocity(right_rad)
    
        # 9) Publish for monitoring.
        self.left_pub.publish(Float32(data=left_rad))
        self.right_pub.publish(Float32(data=right_rad))
    
        self.get_logger().debug(
            f"v={v_cmd:.2f}, w={w_cmd:.2f} → L={left_mph:.2f}mph, R={right_mph:.2f}mph"
        )
    

def main(args=None):
    rclpy.init(args=args)
    robot  = Robot()
    lm     = robot.getDevice('left wheel motor')
    rm     = robot.getDevice('right wheel motor')
    localize = None
    img_h, img_w = 800, 1200

    node = RobotController(lm, rm, localize, img_h, img_w)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
