print("open file")
from controller import Robot
import cv2
import numpy as np
import math
import struct
from RRTStar import RRTStar

import rclpy
from rclpy.node import Node

class GlobalNavController(Node):
    """
        Global Navigation Controller for a robot in a simulation environment.

        Variables:
            - camera: Camera device for capturing images.
            - display: Display device for showing images.
            - emitter: Emitter device for sending messages.
            - receiver: Receiver device for receiving messages.

    """
    def __init__(self):
        #super(GlobalNavController, self).__init__("global_nav_controller")
        super().__init__('global_nav_controller')
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize devices
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
#        self.display = self.getDevice('display')
        
       # self.left_motor = self.getDevice('left wheel motor')
        # self.right_motor = self.getDevice('right wheel motor')
        # self.left_motor.setPosition(float('inf'))
        # self.right_motor.setPosition(float('inf'))
        # self.left_motor.setVelocity(0.0)
        # self.right_motor.setVelocity(0.0)
        self.imu = self.robot.getDevice('IMU')
        self.imu.enable(self.timestep)
        # self.lidar = self.getDevice('lidar')
        # self.lidar.enable(self.timestep)
        

        self.com_val = None
        
        self.robot_pos = None
        self.next_goal = (4, 4)  # Hardcoded goal in world coordinates (x, y)
        self.path = None

    def create_obstacle_map(self, frame):
        """
            Generate obstacle map from camera image.
            
            Args:
                frame: The camera image frame.

            Returns:
                obstacle_map: A binary map where 1 represents obstacles and 0 represents free space.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        obstacle_map = (gray < 50).astype(int)  # 1 for obstacles, 0 for free space
        return obstacle_map
        
    

    def world_to_pixel(self, world_x, world_y):
        """Convert world coordinates to pixel coordinates (640x640 image).
        
            Args:
                world_x: X coordinate in world coordinates.
                world_y: Y coordinate in world coordinates.
            Returns:
                pixel coordinates (j, i) in the image.
        """
        pixel_j = int((world_x + 5) * 639 / 10)
        pixel_i = int((5 - world_y) * 639 / 10)
        return (pixel_j, pixel_i)

    def plan_path(self, start_pixel, goal_pixel, obstacle_map):
        """ Plan a path using RRT*.
            Description:


            Args:
                start_pixel: Starting pixel coordinates (j, i).
                goal_pixel: Goal pixel coordinates (j, i).
                obstacle_map: Binary map of obstacles.


            Returns:
                Path: A list of pixel coordinates representing the planned path.
        
        """
        margin = 100
        rand_area = [min(start_pixel[0], goal_pixel[0]) - margin,
                     max(start_pixel[0], goal_pixel[0]) + margin]
        rrt = RRTStar(start=start_pixel, goal=goal_pixel, obstacle_list=obstacle_map,
                      rand_area=rand_area, max_iter=300)
        return rrt.planning(animation=False)

    def calculate_speed(self, path, step_dist=5.0):
        """
            Calculate speed based on path segment.
        
            Args: 
                path: List of pixel coordinates representing the path.

            Returns:
                speed: Calculated speed based on the path segment.
        """
        if path and len(path) > 1:
            p1, p2 = path[0], path[1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0:
                return min(step_dist / dist * 5.0, 5.0)  # Simulated speed in m/s
        return 0.0

    def run(self):
        """
            Main loop for the robot controller.
        """
        print("Inside run")
        
        while self.robot.step(self.timestep) != -1:
            # Get camera image
            image = np.array(self.camera.getImageArray(), dtype=np.uint8)
            obstacle_map = self.create_obstacle_map(image)
            
            # Get robot position
            
            pos = self.robot.get_position()
            self.robot_pos = (pos[0], pos[1])
            robot_pixel = self.world_to_pixel(self.robot_pos[0], self.robot_pos[1])
            goal_pixel = self.world_to_pixel(self.next_goal[0], self.next_goal[1])
            
            # Plan path
            self.path = self.plan_path(robot_pixel, goal_pixel, obstacle_map)
            
            # Calculate speed (simulated, not used due to commented motors)
            speed = self.calculate_speed(self.path)
            
            # Create display image
            display_image = np.zeros((640, 640, 3), dtype=np.uint8)
            display_image[obstacle_map == 1] = [0, 0, 0]  # Black for obstacles
            display_image[obstacle_map == 0] = [255, 255, 255]  # White for free space
            
            if self.path:
                for i in range(len(self.path) - 1):
                    cv2.line(display_image, tuple(self.path[i]), tuple(self.path[i + 1]),
                            (0, 255, 0), 2)  # Green path
            cv2.circle(display_image, robot_pixel, 5, (255, 0, 0), -1)  # Red robot
            cv2.circle(display_image, goal_pixel, 5, (0, 255, 255), -1)  # Yellow goal
            
            # Publish map via display
            data = display_image.tobytes()
            self.display.imageNew(640, 640, data, 'RGB')
            
            
            # Subscribe to com_val
            if self.receiver.getQueueLength() > 0:
                message = self.receiver.getData()
                self.com_val = struct.unpack('f', message)[0]
                self.receiver.nextPacket()
                print(f"Received com_val: {self.com_val}")




def main():
    
    
    rclpy.init()

    # Instantiate and run the controller
    controller = GlobalNavController()
    controller.run()
    

if __name__ == "__main__":
    main()