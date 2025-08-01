# Fusion-2.py
"""
File: Fusion-2.py - Fused Controller for Robot Navigation

This file implements a fused controller for a robot that integrates
computer vision, LIDAR, and GPS data to navigate autonomously.
It uses YOLO for object detection, RRT* for path planning,
and various sensors to maintain localization and control.
It also includes methods for detecting lane markings,
    adjusting goal points, and selecting next goals based on drivable areas.

    Required Libraries:
        math - Mathematical functions
        numpy - Numerical operations
        cv2 - OpenCV for image processing
        time - Time functions
        rclpy - ROS 2 Python client library
        sensor_msgs - ROS 2 message types for sensors
        geometry_msgs - ROS 2 message types for geometry
        nav_msgs - ROS 2 message types for navigation
        cv_bridge - ROS 2 bridge for OpenCV images
        controller - Webots controller API
        ultralytics - YOLO model for object detection   
        tf_transformations - ROS 2 transformations
        tf2_ros - ROS 2 TF2 for broadcasting transforms
        RRTStar - RRT* path planning algorithm
        RobotLocalization - Robot localization module
        RRTPathToGPSConverter - Converts RRT paths to GPS coordinates
        DrivableAreaController - Detects drivable areas
        WaypointNavigator - Navigates waypoints

Methods:
    detect_lane1 - Detects lane markings in an image
    adjust_goal_point - Adjusts goal point to stay within drivable area
    get_next_goal_lane - Computes next goal point based on lane detection
    select_goal_point - Selects the best goal point within the drivable area
    yolo_to_obstacle_map - Converts YOLO detections to obstacle map
    yolo_track_to_obstacle_map - Converts YOLO track results to obstacle map
    pad_goal_from_obstacles - Pads goal point away from obstacles
    lidar_to_occupancy - Converts LIDAR scan to occupancy grid
    gps_to_pixel - Converts GPS coordinates to pixel coordinates

History 
     5/10/2025 AI - Updated hyper parameters for path planning and next goal
     5/26/2025 AI - Updated goal selection to use drivable area and lane detection
"""
import math
import numpy as np
import cv2
import time

from sensor_msgs.msg import Image, LaserScan, Imu, NavSatFix
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge

from controller import Robot
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

import tf_transformations
from tf2_ros import TransformBroadcaster

from RRTStar import RRTStar
from RobotLocalization import RobotLocalization
#from CostMapPlanner import update_cost_map
from RobotController import WebotsRosController
from RRTPathToGPSConverter import RRTPathToGPSConverter
from DrivableAreaController import DrivableAreaDetector
from WaypointNavigator import WaypointNavigator


model = YOLO("/home/ubuntu/worlds/IGVC25/controllers/Fusion -2/best.pt")
print("YOLO model loaded.")


import torch
import numpy as np
import utm



def detect_lane(image,
                 roi_vertical_start=0.5,
                 canny_thresh1=50, canny_thresh2=150,
                 hough_rho=1, hough_theta=np.pi/180,
                 hough_thresh=50, min_line_len=50, max_line_gap=10):
    """
    Detect white/yellow lane markings in a forward-facing image.

    Args:
        image:     input image (BGR)
        roi_vertical_start: fraction of image height to start ROI
        canny_thresh1: Canny edge detection threshold 1
        canny_thresh2: Canny edge detection threshold 2
        hough_rho: Hough transform rho resolution
        hough_theta: Hough transform theta resolution
        hough_thresh: Hough transform threshold
        min_line_len: minimum line length to be considered a line
        max_line_gap: maximum gap between line segments to be connected

    Returns:
      lane_mask: single-channel binary mask of lane pixels across the relevant ROI.
                 Note: This implementation only processes the ROI [y0:, :]
                 and returns a mask of that size. It should be expanded
                 to a full-size mask if needed for consistency with the
                 drivable function. Let's update this dummy to return
                 a full size mask for better compatibility.
      lines:     list of line segments [[x1,y1,x2,y2],…] in full image coords or None.
    """
    H, W = image.shape[:2]
    # 1) Restrict to lower half (road ahead) - Note: This ROI is different from the drivable area ROI
    y0 = int(H * roi_vertical_start)
    # Process the full image or a slightly larger area if your actual detect_lane does
    # For this dummy, let's process the lower half as it's written, but ensure the mask
    # is returned as full size with the detected pixels in the lower part.
    process_roi_y0 = int(H * 0.4) # Let's process from 40% down for dummy

    process_roi = image[process_roi_y0:, :]


    # 2) Color threshold in HSV
    hsv = cv2.cvtColor(process_roi, cv2.COLOR_BGR2HSV)
    # Adjusted color ranges slightly for potentially better detection in diverse conditions
    white = cv2.inRange(hsv, (0, 0, 190), (180, 30, 255)) # Higher min V for white
    yellow = cv2.inRange(hsv, (15, 80, 80), (45, 255, 255)) # Wider yellow range
    mask = cv2.bitwise_or(white, yellow)

    # 3) Morphological closing to seal corners, then blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)

    # 4) Edge detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 5) Probabilistic Hough
    lines = cv2.HoughLinesP(edges,
                            rho=hough_rho,
                            theta=hough_theta,
                            threshold=hough_thresh,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # lines are in process_roi coords; shift them back to full image
    if lines is not None:
        lines[:, 0, 1] += process_roi_y0
        lines[:, 0, 3] += process_roi_y0

    # 6) Create full image size lane mask
    lane_mask_full = np.zeros((H, W), dtype=np.uint8)
    lane_mask_full[process_roi_y0:, :] = closed # Place processed mask back into full image size


    # Dummy logic to simulate different lane detection cases based on x position of lines
    # In a real scenario, this would be a result of your actual detection logic
    detected_two = False
    detected_one_left = False
    detected_one_right = False

    if lines is not None:
        leftish_lines = [l for l in lines[:, 0] if ((l[0] + l[2]) / 2 < W // 2 - 50)] # Avg x left of center
        rightish_lines = [l for l in lines[:, 0] if ((l[0] + l[2]) / 2 > W // 2 + 50)] # Avg x right of center

        if len(leftish_lines) > 0 and len(rightish_lines) > 0:
            # Check if lines cover a reasonable vertical range in the drivable ROI
            min_y_left = min([min(l[1], l[3]) for l in leftish_lines])
            max_y_left = max([max(l[1], l[3]) for l in leftish_lines])
            min_y_right = min([min(l[1], l[3]) for l in rightish_lines])
            max_y_right = max([max(l[1], l[3]) for l in rightish_lines])

            # Simple check: do lines span enough vertical distance within a plausible ROI range?
            # This is a very rough heuristic! Your drivable function counts rows instead.
            # Let's make this dummy mode setting simpler based on a flag.
            pass # Logic moved to example calls

    

    return lane_mask_full, lines # Return full mask and lines

def detect_lane1(image,
                roi_vertical_start=0.5,
                canny_thresh1=50, canny_thresh2=150,
                hough_rho=1, hough_theta=np.pi/180,
                hough_thresh=50, min_line_len=50, max_line_gap=10):
    """
    Detect white/yellow lane markings in a forward-facing image.

    Args:
        image:    input image (BGR)
        roi_vertical_start: fraction of image height to start ROI
        canny_thresh1: Canny edge detection threshold 1
        canny_thresh2: Canny edge detection threshold 2
        hough_rho: Hough transform rho resolution
        hough_theta: Hough transform theta resolution
        hough_thresh: Hough transform threshold
        min_line_len: minimum line length to be considered a line
        max_line_gap: maximum gap between line segments to be connected

    Returns:
      lane_mask: single-channel binary mask of lane pixels
      lines:    list of line segments [[x1,y1,x2,y2],…] or None
    """
    H, W = image.shape[:2]
    # 1) Restrict to lower half (road ahead)
    y0 = int(H * roi_vertical_start)
    roi = image[y0:, :]

    # 2) Color threshold in HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 25, 255))
    yellow = cv2.inRange(hsv, (20, 100, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)

    # 3) Morphological closing to seal corners, then blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)

    # 4) Edge detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 5) Probabilistic Hough
    lines = cv2.HoughLinesP(edges,
                            rho=hough_rho,
                            theta=hough_theta,
                            threshold=hough_thresh,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # lines are in ROI coords; shift them back to full image
    if lines is not None:
        lines[:, 0, 1] += y0
        lines[:, 0, 3] += y0

    # 6) Expand mask back to full image size (for masking later)
    lane_mask = np.zeros((H, W), dtype=np.uint8)
    lane_mask[y0:, :] = closed

    return lane_mask, lines





def adjust_goal_point(original_goal, lane_mask, drivable_mask, robot_xy, pullback_ratio=0.7):
    """
    Adjust goal point to stay within drivable area while pulling it closer to the robot.
    
    Args:
        original_goal (tuple): Initial goal coordinates (x, y)
        lane_mask: Binary lane mask (white=lane)
        drivable_mask: Binary drivable area mask (white=drivable)
        robot_xy (tuple): Robot's current position (x, y)
        pullback_ratio: How much to pull back towards robot (0.0-1.0)
        
    Returns:
        tuple: Adjusted (x, y) coordinates
    """
    robot_x, robot_y = robot_xy
    orig_x, orig_y = original_goal
    
    # Calculate vector from robot to original goal
    dx = orig_x - robot_x
    dy = orig_y - robot_y
    
    # Create pullback point closer to robot
    pullback_x = robot_x + dx * pullback_ratio
    pullback_y = robot_y + dy * pullback_ratio
    
    # Find drivable points near the pullback location
    ys, xs = np.where(drivable_mask > 0)
    if len(xs) == 0:
        return original_goal  # No adjustment possible
    
    # Convert to numpy array for efficient calculations
    drivable_pts = np.column_stack((xs, ys))
    
    # Find nearest drivable point to pullback location
    distances = np.linalg.norm(drivable_pts - [pullback_x, pullback_y], axis=1)
    closest_idx = np.argmin(distances)
    
    return (drivable_pts[closest_idx][0], drivable_pts[closest_idx][1])


def get_next_goal_lane(
    robot_xy,
    robot_heading,
    obstacle_map,
    lane_mask,
    drivable_mask, 
    forward_dist=200,
    roi_width=100,
    roi_height=50,
    min_area=10,
    single_lane_margin=3
):
    """
    Compute the next goal point:
     - If two lanes are detected in the lane_mask ROI, goal = midpoint between lanes.
     - If one lane is detected, goal = farthest right within drivable area (with margin).
     - Else, find the largest obstacle in obstacle_map ROI (masked by lane_mask) and goal = its centroid.

    Returns:
        (x_goal, y_goal) in pixel coords, or None if nothing found.
    """
    """
    1) TWO lanes → midpoint between them.
    2) ONE lane  → middle of drivable_mask in ROI.
    3) ZERO lanes → largest obstacle in obstacle_map.
    """
    H, W = obstacle_map.shape
    # 1) ROI in front of robot
    cx = int(robot_xy[0] + math.cos(robot_heading)*forward_dist)
    cy = int(robot_xy[1] + math.sin(robot_heading)*forward_dist)
    x0 = max(0, min(W - roi_width,  cx - roi_width//2))
    y0 = max(0, min(H - roi_height, cy - roi_height//2))
    x1, y1 = x0 + roi_width, y0 + roi_height

    lane_roi      = lane_mask[y0:y1, x0:x1]
    driv_roi      = drivable_mask[y0:y1, x0:x1]
    obstacle_roi  = obstacle_map[y0:y1, x0:x1]

    def find_centroids(mask):
        cnts,_ = cv2.findContours((mask>0).astype(np.uint8),
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            A = cv2.contourArea(c)
            if A<min_area: continue
            M = cv2.moments(c)
            if M['m00']==0: continue
            cx_r = M['m10']/M['m00']
            cy_r = M['m01']/M['m00']
            out.append((cx_r, cy_r, A, c))
        return out

    lane_cents = find_centroids(lane_roi)

    # --- 1) TWO lanes → midpoint -------------------
    if len(lane_cents) >= 2:
        print("2 Lanes")
        lane_cents.sort(key=lambda x: x[2], reverse=True)  # Sort by area/confidence
    
        # Get lane centroids in ROI coordinates
        xA, yA, _, _ = lane_cents[0]
        xB, yB, _, _ = lane_cents[1]
    
        # Calculate lateral midpoint between lanes
        mid_x = (xA + xB) / 2
    
        # Project forward using robot heading and forward_dist
        proj_y = forward_dist  # Use full forward distance for projection
    
        # Convert to image coordinates (assuming ROI is centered at (cx, cy))
        goal_x = x0 + mid_x
        goal_y = cy  # Maintain original forward distance center
    
        # Alternative: If needing to add to robot's original position
        # goal_x = robot_x + math.cos(robot_heading) * forward_dist + mid_x
        # goal_y = robot_y + math.sin(robot_heading) * forward_dist
    
        return float(goal_x), float(goal_y)

    # --- 2) ONE lane → middle of drivable_mask ----
    if len(lane_cents) == 1:
        # fetch bias from parameters
        print("One Lane")
        params =  0.7#self.get_parameters(['drivable_bias'])
        bias = 0.7# params[0].value

        # crop drivable area
        driv = drivable_mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(driv > 0)
        if len(xs) > min_area:
            left_edge  = float(xs.min())
            right_edge = float(xs.max())

            # weighted midpoint
            mid_local_x = left_edge + bias * (right_edge - left_edge)
            front_local_y = float(ys.min())
            print(f"Initial Goal: {float(x0 + mid_local_x)} Y: {float(y0 + front_local_y)}")
            
            
            adjusted_goal = adjust_goal_point((float(x0 + mid_local_x), 
                float(y0 + front_local_y)) , lane_mask, drivable_mask, robot_xy)
            adjusted_goal = adjusted_goal
            print(f"Adjusted Goal: {adjusted_goal}")
         
            return float(adjusted_goal[0]), float(adjusted_goal[1]) # Step 3: Fall back to original one-lane logic if not enough drivable space
        else: # ELSE TO len(xs) > min_area:
            goal_x, goal_y = select_goal_point( lane_mask, drivable_mask,
                robot_x=robot_xy[0], robot_y=robot_xy[1],
                robot_heading=robot_heading, forward_dist=forward_dist,
                roi_width=roi_width,roi_height=roi_height)
            if goal_x is not None and  goal_x != (None, None):
                print(f"goal x {goal_x} ")
                if isinstance(goal_x, tuple):
                    # Assuming the numeric value is the first element of the tuple.
                    goal_x = goal_x[0]
                    goal_y = goal_y[0] if isinstance(goal_y, tuple) else goal_y
                print(f"Found goal at ({goal_x}, {goal_y})")
                return float(goal_x), float(goal_y)
            else:
                print("No valid goal point found")
    
    ys, xs = np.nonzero(driv_roi > 0)
    """
    if len(xs) > min_area:
        front_y = ys.min()
        goal_x = (xs.min() + xs.max()) / 2  # Midpoint of drivable area
        goal_x = np.clip(goal_x, single_lane_margin, roi_width - single_lane_margin)
        goal_y = float(y0 + front_y)

        return float(x0 + goal_x), float(y0 + goal_y)
    """
    # --- 3) ZERO lanes → obstacle fallback -------
    obs_cents = find_centroids(obstacle_roi)
    if obs_cents:
        obs_cents.sort(key=lambda x: x[2], reverse=True)
        xo, yo, _, _ = obs_cents[0]
        return float(x0 + xo), float(y0 + yo)
    
    
    ys, xs = np.nonzero(driv_roi > 0)
    
   
    
    if len(xs) > min_area:
        print("Using Drivable Area Mid Point")
        # Compute the centroid of the largest drivable region
        mid_x = (xs.min() + xs.max()) / 2
        mid_y = (ys.min() + ys.max()) / 2
        return float(x0 + mid_x), float(y0 + mid_y)
    
    
    
    # --- 5)  fallback find midpoint of drivable area at half of forward_dist ------
    half_forward_dist = forward_dist // 2
    cx_half = int(robot_xy[0] + math.cos(robot_heading) * half_forward_dist)
    cy_half = int(robot_xy[1] + math.sin(robot_heading) * half_forward_dist)

    x0_half = max(0, min(W - roi_width, cx_half - roi_width // 2))
    y0_half = max(0, min(H - roi_height, cy_half - roi_height // 2))
    x1_half, y1_half = x0_half + roi_width, y0_half + roi_height

    driv_half_roi = drivable_mask[y0_half:y1_half, x0_half:x1_half]
    ys_half, xs_half = np.nonzero(driv_half_roi > 0)

    if len(xs_half) > min_area:
        print("Using Half the hight and drivable area")
        mid_x_half = (xs_half.min() + xs_half.max()) / 2
        mid_y_half = (ys_half.min() + ys_half.max()) / 2
        return float(x0_half + mid_x_half), float(y0_half + mid_y_half)
    
    
    # --- 6)  fallback → midpoint of drivable area at 1/4 forward_dist ------
    quarter_forward_dist = forward_dist // 4
    cx_quarter = int(robot_xy[0] + quarter_forward_dist)
    cy_quarter = int(robot_xy[1])  # Ignore heading, stay in same Y axis

    x0_quarter = max(0, min(W - roi_width, cx_quarter - roi_width // 2))
    y0_quarter = max(0, min(H - roi_height, cy_quarter - roi_height // 2))
    x1_quarter, y1_quarter = x0_quarter + roi_width, y0_quarter + roi_height

    driv_quarter_roi = drivable_mask[y0_quarter:y1_quarter, x0_quarter:x1_quarter]
    ys_quarter, xs_quarter = np.nonzero(driv_quarter_roi > 0)

    if len(xs_quarter) > min_area:
        print("Using a quarter of the drivable area")
        mid_x_quarter = (xs_quarter.min() + xs_quarter.max()) / 2
        mid_y_quarter = (ys_quarter.min() + ys_quarter.max()) / 2
        return float(x0_quarter + mid_x_quarter), float(y0_quarter + mid_y_quarter)

    
    print("no goal point")
    # nothing found
    return None
    
    

def select_goal_point( lane_mask_raw, drivable_mask, robot_x, robot_y, robot_heading, 
                     forward_dist, roi_width, roi_height):
    """
    Selects the best goal point within the drivable area considering lane boundaries.
    
    Args:
        bgr: Original color image (for dimensions)
        lane_mask_raw: Binary lane mask (white=lane)
        drivable_mask: Binary drivable area mask (white=drivable)
        robot_x, robot_y: Robot position in image coordinates
        robot_heading: Robot heading in radians
        forward_dist: Distance forward to look for goal
        roi_width, roi_height: Size of search region
        
    Returns:
         (goal_x, goal_y) coordinates or (None, None) if no valid goal found
        str: Source of goal point ("Drivable", "Lane", or "None")
    """
    H, W = drivable_mask.shape[:2]
    
    # Calculate center of search region
    cx = int(robot_x + math.cos(robot_heading) * forward_dist)
    cy = int(robot_y + math.sin(robot_heading) * forward_dist)
    
    # Define ROI boundaries (clamped to image)
    x0 = max(cx - roi_width//2, 0)
    x1 = min(cx + roi_width//2, W)
    y0 = max(cy - roi_height//2, 0)
    y1 = min(cy + roi_height//2, H)
    
    # Initialize return values
    goal_x, goal_y = None, None
    source = "None"
    
    if y0 >= y1 or x0 >= x1:
        return (None, None), "Invalid ROI"
    
    # Get ROI slices
    drivable_roi = drivable_mask[y0:y1, x0:x1]
    lane_roi = lane_mask_raw[y0:y1, x0:x1]
    
    # Find all drivable points in ROI
    drivable_pts = np.column_stack(np.where(drivable_roi > 0))
    if len(drivable_pts) > 0:
        # Convert to absolute coordinates
        drivable_pts[:, 0] += y0  # y-coordinate
        drivable_pts[:, 1] += x0  # x-coordinate
        
        # Find point closest to center of ROI
        center_pt = np.array([cy, cx])
        distances = np.linalg.norm(drivable_pts - center_pt, axis=1)
        closest_idx = np.argmin(distances)
        goal_y, goal_x = drivable_pts[closest_idx]
        source = "Drivable"
    else:
        # Fallback to lane points if no drivable area
        lane_pts = np.column_stack(np.where(lane_roi > 0))
        if len(lane_pts) > 0:
            lane_pts[:, 0] += y0
            lane_pts[:, 1] += x0
            distances = np.linalg.norm(lane_pts - center_pt, axis=1)
            closest_idx = np.argmin(distances)
            goal_y, goal_x = lane_pts[closest_idx]
            source = "Lane"
    print(f"source {source}")
    return  goal_x, goal_y #, source


    
def yolo_to_obstacle_map(detections, image_shape, conf_thresh=0.5):
    """
    Converts YOLO detections (list of [x1,y1,x2,y2,conf,class_id])
    into a binary obstacle map of shape image_shape (H, W).
    Any pixel inside a kept bbox is marked 1.
    """
    H, W = image_shape
    obs_map = np.zeros((H, W), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_thresh:
            continue
        # clamp coords to image bounds
        x1i = max(0, min(W-1, int(x1)))
        y1i = max(0, min(H-1, int(y1)))
        x2i = max(0, min(W-1, int(x2)))
        y2i = max(0, min(H-1, int(y2)))
        # fill the rectangle in the obstacle map
        cv2.rectangle(obs_map,
                      (x1i, y1i),
                      (x2i, y2i),
                      color=1,
                      thickness=-1)

    return obs_map








def yolo_track_to_obstacle_map(res, image_shape, conf_thresh=0.0, classes=None, debug=False):
    """
    Convert YOLOv8 track() results into a binary obstacle map.
    Any bbox with conf >= conf_thresh and (cls in classes) will be filled.

    Args:
      res:          Results or list of Results from model.track(..., persist=True)
      image_shape:  (H, W) of your input
      conf_thresh:  include anything above this confidence (set to 0.0 to debug)
      classes:      optional set of class IDs to include
      debug:        if True, print and visualize each box

    Returns:
      obs_map: uint8 H×W, with 255 inside each kept bbox
    """
    H, W = image_shape
    obs_map = np.zeros((H, W), dtype=np.uint8)

    results = res if isinstance(res, (list,tuple)) else [res]
    for r in results:
        # r.boxes.xyxy is a tensor Nx4, r.boxes.conf & r.boxes.cls are Nx1
        bboxes = r.boxes.xyxy.cpu().numpy()   # shape (N,4)
        confs  = r.boxes.conf.cpu().numpy()   # shape (N,)
        clss   = r.boxes.cls.cpu().numpy().astype(int)  # shape (N,)
        for (x1,y1,x2,y2), conf, cls in zip(bboxes, confs, clss):
            if conf < conf_thresh:
                continue
            if classes is not None and cls not in classes:
                continue

            # convert to int pixel coords & clamp
            x1i, y1i = max(0,int(x1)), max(0,int(y1))
            x2i, y2i = min(W-1,int(x2)), min(H-1,int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue

            # draw filled rectangle with value=255
            cv2.rectangle(obs_map, (x1i,y1i), (x2i,y2i), color=255, thickness=-1)

            if debug:
                print(f"DBG: box cls={cls} conf={conf:.2f} -> roi {x1i,y1i,x2i,y2i}")
                cv2.rectangle(vis, (x1i,y1i), (x2i,y2i), (0,0,255), 2)

    if debug:
        print("Obstacle pixels:", np.count_nonzero(obs_map))
    return obs_map


def pad_goal_from_obstacles(x_goal, y_goal, obstacle_map, pad_px=40):
    """
    If there are any obstacle pixels within a pad_px×pad_px box around (x_goal,y_goal),
    compute the centroid of those pixels and push the goal *away* by exactly pad_px pixels.
    """
    H, W = obstacle_map.shape
    # Define a small square ROI around the goal
    x0 = max(0, int(x_goal) - pad_px)
    x1 = min(W, int(x_goal) + pad_px)
    y0 = max(0, int(y_goal) - pad_px)
    y1 = min(H, int(y_goal) + pad_px)
    roi = obstacle_map[y0:y1, x0:x1]
    ys, xs = np.nonzero(roi > 0)
    if len(xs) == 0:
        return x_goal, y_goal  # no obstacles, no shift

    # Centroid of obstacles in the ROI (in full‐image coords)
    cx_obs = x0 + xs.mean()
    cy_obs = y0 + ys.mean()

    # Vector from obstacle towards goal
    vx = x_goal - cx_obs
    vy = y_goal - cy_obs
    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        # goal dead‐on top of obstacles: just move straight back along Y
        return x_goal, y_goal - pad_px

    # Normalize and scale to pad_px
    vx *= pad_px / norm
    vy *= pad_px / norm

    return x_goal + vx, y_goal + vy

def check_obj_distances(distances, threshold=200):
    """
    Check if any distance measurement is below a threshold (in pixels).
    
    Args:
        distances: List of distance measurements
        threshold: Distance threshold in pixels (default: 200)
    
    Returns:
        Tuple of (boolean, list):
        - True if any distance < threshold
        - List of indices where distance < threshold
    """
    warnings = []
    for i, dist in enumerate(distances):
        if dist < threshold:
            warnings.append(i)
    
    return (len(warnings) > 1)







def lidar_to_occupancy(ranges, angle_min, angle_max, 
                      grid_size=400, meters_per_pixel=0.1,
                      max_range=10.0, debug=False):
    """
    Convert LiDAR ranges to binary obstacle map similar to YOLO version
    
    Args:
        ranges: List of distance measurements (meters)
        angle_min/max: Start/end angles of scan (radians)
        grid_size: Size of square output map (pixels)
        meters_per_pixel: Map resolution (m/pixel)
        max_range: Ignore measurements beyond this distance
        debug: Print debug info
        
    Returns:
        obs_map: uint8 grid_size×grid_size, 255=obstacle
    """
    # Initialize empty map (robot at center)
    obs_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    center = grid_size // 2
    
    # Convert polar to Cartesian and mark obstacles
    for i, dist in enumerate(ranges):
        if dist > max_range or np.isinf(dist):
            continue
            
        # Calculate angle for this measurement
        angle = angle_min + i*(angle_max - angle_min)/len(ranges)
        
        # Convert to grid coordinates
        x = dist * np.cos(angle) / meters_per_pixel
        y = dist * np.sin(angle) / meters_per_pixel
        u = int(center + x)
        v = int(center - y)  # Flip Y-axis for image coordinates
        
        # Mark obstacle if within bounds
        if 0 <= u < grid_size and 0 <= v < grid_size:
            cv2.circle(obs_map, (u, v), radius=2, color=255, thickness=-1)
            
            if debug:
                print(f"LIDAR @ ({u},{v}) - {dist:.2f}m")

    if debug:
        cv2.imshow("Debug Lidar", obs_map)
        cv2.waitKey(1)
        
    return obs_map

def get_obstacle_info(res, image_shape, pixel_thresh=400):
    """
    Args:
        res: YOLOv11n `model.track().
        image_shape: tuple (H, W) of image dimensions.
        pixel_thresh: vertical pixel distance from bottom up (default 400).

    Returns:
        count: number of valid obstacle boxes within the given pixel_thresh.
        Distances: list of vertical distances (in pixels) from the robot origin to the box centers.
    """
    H, W = image_shape
    origin_y = H
    results = res if isinstance(res, (list, tuple)) else [res]
   
    counter = 0
    distances = []

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = box.tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            vertical_dist = origin_y - cy  
            if vertical_dist > 0 and vertical_dist <= pixel_thresh:
                counter += 1
                distances.append(vertical_dist)

    return counter, distances


def gps_to_pixel(lat, lon, navigator):
        """Convert GPS to pixel coordinates using your UTM setup"""
        easting, northing, _, _ = utm.from_latlon(lat, lon)
        dx = easting - navigator.origin_easting
        dy = northing - navigator.origin_northing
        return (
            int(dx * navigator.pixels_per_meter),
            int(-dy * navigator.pixels_per_meter))  





def get_safe_retreat(occ_map, fov_mask, lane_mask,
                     forward_pixels=200, roi_width=400):
    """
    Returns the absolute (row, col) in the full image that maximizes
    occ_map * fov_mask * lane_mask within a 200px window in front.
    """
    H, W = occ_map.shape

    # Crop last `forward_pixels` rows
    occ_roi = occ_map[H - forward_pixels : H, :]
    fov_roi = fov_mask[H - forward_pixels : H, :]
    lane_roi = lane_mask[H - forward_pixels : H, :]

    # Center‐horizontal crop
    center = W // 2
    half = roi_width // 2
    c0 = max(center - half, 0)
    c1 = min(center + half, W)

    occ_roi   = occ_roi[:, c0:c1]
    fov_roi   = fov_roi[:, c0:c1]
    lane_roi  = lane_roi[:, c0:c1]

    # Combined safety score
    score = occ_roi * fov_roi * lane_roi

    # Best pixel in ROI
    rel_row, rel_col = np.unravel_index(np.argmax(score), score.shape)

    # Convert back to full‐image coords
    abs_row = (H - forward_pixels) + rel_row
    abs_col = c0 +       rel_col

    # Sanity check
    assert H - forward_pixels <= abs_row < H, (
        f"abs_row={abs_row} outside [{H-forward_pixels}, {H})"
    )

    return abs_row, abs_col


def get_safe_retreat_vec(occ_map, fov_mask, lane_mask,
                         forward_pixels=200, roi_width=400):
    """
    Returns (dx, dy) in pixels from the vehicle (bottom center):
      - dy > 0 means "forward" up to `forward_pixels`
      - dx > 0 means to the right
    """
    H, W = occ_map.shape

    # Same ROI cropping
    occ_roi = occ_map[H - forward_pixels : H, :]
    fov_roi = fov_mask[H - forward_pixels : H, :]
    lane_roi = lane_mask[H - forward_pixels : H, :]

    center = W // 2
    half   = roi_width // 2
    c0 = max(center - half, 0)
    c1 = min(center + half, W)

    occ_roi   = occ_roi[:, c0:c1]
    fov_roi   = fov_roi[:, c0:c1]
    lane_roi  = lane_roi[:, c0:c1]

    score = occ_roi * fov_roi * lane_roi
    rel_row, rel_col = np.unravel_index(np.argmax(score), score.shape)

    # Pixel‐space vector from bottom‐center:
    #   vehicle at (row=H-1, col=center)
    #   target at (row=(H-forward_pixels)+rel_row, col=c0+rel_col)
    target_row = (H - forward_pixels) + rel_row
    target_col = c0 + rel_col

    dy = (H - 1) - target_row      # positive = backward in image coords
    dx = target_col - center       # positive = right

    # But we want "forward" positive, so flip the sign of dy:
    dy = -dy

    return dx, dy


def convert_to_world(point, grid_resolution=0.1, origin=(0, 0)):
    """
    Converts a grid‐pixel vector (dx, dy) into world meters.
    """
    x_pix, y_pix = point
    x_m = x_pix * grid_resolution + origin[0]
    y_m = y_pix * grid_resolution + origin[1]
    return x_m, y_m



def pad_goal_from_lines(x_goal, y_goal, lines, pad_px=40):
    """
    If any endpoints of line segments in `lines` fall within a pad_px×pad_px window
    around (x_goal, y_goal), compute their centroid and push the goal
    *away* by exactly pad_px pixels along the vector from that centroid.

    Parameters
    ----------
    x_goal : float
      Goal column (horizontal) index.
    y_goal : float
      Goal row (vertical) index.
    lines : array-like
      Array (or list) of line segments, where each line is [x1, y1, x2, y2].
      (The array can be of shape (N,4) or (N,1,4); if the latter, it will be squeezed.)
    pad_px : int
      The minimum distance (in pixels) enforced from any line endpoint.

    Returns
    -------
    (x_new, y_new) : tuple of floats
      The adjusted goal, guaranteed to lie at least pad_px away from the nearby line endpoints.
    """
    # If no lines were found, return the goal unchanged.
    if lines is None or len(lines) == 0:
        return x_goal, y_goal

    # Ensure lines is a NumPy array and squeeze in case of shape (N,1,4)
    lines = np.asarray(lines)
    if lines.ndim == 3:
        lines = np.squeeze(lines, axis=1)  # now shape (N, 4)

    endpoints = []
    # Loop over each line and check both endpoints.
    for line in lines:
        x1, y1, x2, y2 = line
        # Check if the first endpoint is within the pad window
        if (x_goal - pad_px <= x1 <= x_goal + pad_px) and (y_goal - pad_px <= y1 <= y_goal + pad_px):
            endpoints.append((x1, y1))
        # Check if the second endpoint is within the pad window
        if (x_goal - pad_px <= x2 <= x_goal + pad_px) and (y_goal - pad_px <= y2 <= y_goal + pad_px):
            endpoints.append((x2, y2))

    # 2) If no endpoints are nearby, leave the goal as-is.
    if len(endpoints) == 0:
        return x_goal, y_goal

    # 3) Compute the centroid of the nearby endpoints (full-image coordinates).
    endpoints = np.array(endpoints)  # shape (M, 2)
    cx = endpoints[:, 0].mean()
    cy = endpoints[:, 1].mean()

    # 4) Compute vector from the centroid to the goal.
    vx = x_goal - cx
    vy = y_goal - cy
    dist = math.hypot(vx, vy)

    # 5) If the centroid is essentially on the goal, just move straight “up” (–y)
    if dist < 1e-3:
        return x_goal, y_goal - pad_px

    # 6) Normalize the vector and scale it to pad_px.
    vx *= pad_px / dist
    vy *= pad_px / dist

    # 7) Return the “padded” goal.
    return x_goal + vx, y_goal + vy




class FusedController(Node):
    def __init__(self, robot, camera, lidar, imu, gps, left_motor, right_motor):
        super().__init__('fused_controller')
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())

        self.camera = camera; self.camera.enable(self.timestep)
        self.lidar = lidar;   self.lidar.enable(self.timestep)
        self.imu    = imu;    self.imu.enable(self.timestep)
        self.gps    = gps;    self.gps.enable(self.timestep)

        self.left_motor = left_motor; self.left_motor.setPosition(float('inf'))
        self.right_motor= right_motor;self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(1.0); self.right_motor.setVelocity(1.0)

        #self.bridge = CvBridge()
        #self.br = TransformBroadcaster(self)

        self.create_subscription(Twist,'/cmd_vel', self.cmd_cb, 10)
        self.pub_img  = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.pub_imu  = self.create_publisher(Imu, '/imu/data', 10)
        self.pub_gps  = self.create_publisher(NavSatFix, '/gps/fix', 10)
        self.pub_path = self.create_publisher(Path, '/rrt_path', 10)

        self.linear = 1.0
        self.angular= 0.0
        self.last_cmd = time.time()

    def cmd_cb(self, msg: Twist):
        self.linear  = msg.linear.x
        self.angular = msg.angular.z
        self.last_cmd = time.time()

    def update_motors(self):
        if time.time() - self.last_cmd < 1.0:
            ls = self.linear - self.angular
            rs = self.linear + self.angular
        else:
            ls = rs = 1.0  # fallback forward
        self.left_motor.setVelocity(ls)
        self.right_motor.setVelocity(rs)

    def publish_camera(self, frame):
        i = 1
        #msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        #msg.header.stamp = self.get_clock().now().to_msg()
        #self.pub_img.publish(msg)

    def publish_lidar(self):
        ranges = self.lidar.getRangeImage()
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_link"
        msg.angle_min = 0.0
        msg.angle_max = 2 * math.pi
        msg.angle_increment = 2 * math.pi / len(ranges)
        msg.range_min = self.lidar.getMinRange()
        msg.range_max = self.lidar.getMaxRange()
        msg.ranges = list(ranges)
        self.pub_scan.publish(msg)
        return ranges

    def publish_imu(self):
        q = self.imu.getQuaternion()
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
        self.pub_imu.publish(msg)

    def publish_gps(self):
        gps_values = self.gps.getValues()
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "gps_link"
        msg.latitude = float(gps_values[0])
        msg.longitude = float(gps_values[1])
        msg.altitude = float(gps_values[2])
        self.pub_gps.publish(msg)

    def publish_tf(self, x, y, theta):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, theta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        #self.br.sendTransform(t)

    def publish_rrt_path(self, path_points):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"  # 🔄 changed from "map"
        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = "odom"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_path.publish(msg)

    def plan_rrt(self, obst, start, goal):
        margin = 100
        ra = [min(start[0], goal[0]) - margin, max(start[0], goal[0]) + margin]
        rrt = RRTStar(start, goal, obst, rand_area=ra, max_iter=300)
        return rrt.planning(False)
    
    
def main():
    robot = Robot(); ts = int(robot.getBasicTimeStep())
    camera = robot.getDevice('camera'); camera.enable(ts)
    lidar  = robot.getDevice('LDS-01'); lidar.enable(ts)
    imu    = robot.getDevice('IMU');    imu.enable(ts)
    gps    = robot.getDevice('gps');    gps.enable(100)
    left   = robot.getDevice('left wheel'); left.setPosition(float('inf'))
    right  = robot.getDevice('right wheel'); right.setPosition(float('inf'))

    rclpy.init()
    
    
    
    node = FusedController(robot, camera, lidar, imu, gps, left, right)
    w = camera.getWidth(); h = camera.getHeight()
    localizer = RobotLocalization(ts/1000.0, 0.33, 0.0975, left, right, imu=imu)
    position = gps.getValues()
    latitude = 41.129718 #position[0]  # X-coordinate (latitude)
    longitude = -73.552733  #position[1]  # Y-coordinate (longitude)
    print(f"latitude : {latitude} longitude {longitude}")
    gpsconverter = RRTPathToGPSConverter(latitude, longitude)
    
    # # the known GPS of your Engineering Center NW corner:
    # corner_lat =  42.442123   # for example
    #corner_lon = -76.501234
    #
    # Setup the gps cornate
    navigator = WaypointNavigator(
        yaml_file="waypoints.yaml",
        origin_lat=latitude,
        origin_lon=-longitude,
        pixels_per_meter=10  
    )
    
    
    
  
    
    daselector = DrivableAreaDetector(None, h, w)
    print(f"Width: {w} Height {h}")
    while robot.step(ts) != -1:
        rclpy.spin_once(node, timeout_sec=0.001)
        node.update_motors()

        buf = camera.getImage()
        frame = np.frombuffer(buf, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        node.publish_camera(bgr)

        res = model.track(bgr, persist=True)
        vis = res[0].plot() if res and res[0].boxes else bgr.copy()
        #lane_mask, lines, drivable_mask, num_lanes = detect_lane(bgr)
        #drivable_area = get_drivable_area(yolov2, bgr)
        #overlay_and_display(bgr, drivable_area)
        drivable_area = daselector.process_frame(buf)
        
         # display_overlay=True
        lane_mask_raw_valid, lines_valid = detect_lane(bgr.copy())
        
        if lines_valid is not None:
            for x1,y1,x2,y2 in lines_valid[:,0]:
                cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Perception", lane_mask_raw_valid)
        
        node.publish_lidar()
        node.publish_imu()
        node.publish_gps()

        localizer.update()
        x, y, heading = localizer.get_pose()
        
        
        position = gps.getValues()
        latitude = position[0]  # X-coordinate (latitude)
        longitude = position[1]  # Y-coordinate (longitude)
        
        local = False
        
        #if not local:
        x = (w/2)
        y = y + h- 1
        print(f"X:{x} Y:{y} heading:{heading}")
        
        checklidar = False
        ranges = lidar.getRangeImage()   # returns a Python list or numpy array of floats
        occ = lidar_to_occupancy(
            ranges,
            angle_min=0.0,        # or lidar.getFov() offsets
            angle_max=2*math.pi,
            grid_size=400
        )
        
        
        counter, distances =  get_obstacle_info(res, bgr.shape[:2])
        
        print(f"counter {counter} and distance {distances}")
        
        node.publish_tf(x, y, heading)
        num_on = np.sum(drivable_area == 255)
        num_off = np.sum(drivable_area == 0)
        print(f"drivable area on {num_on} vs off {num_off}")
        #obst = update_cost_map(bgr)
        #bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        fow = 100
        obstacle_map = yolo_track_to_obstacle_map(res, frame.shape[:2],
                                     conf_thresh=0.1,
                                     classes=None)
        if check_obj_distances(distances, 350) and num_on < 20:
            """
            drivable_area = lidar_to_drivable_area(ranges,
            angle_min=0.0,        # or lidar.getFov() offsets
            angle_max=2*math.pi,
            grid_size=400)
            obstacle_map = occ
            x,  y = drive_map.shape[:2]
            y = y // 2
            x = x // 2
            """
            #obstacle_map = occ
            if occ is not None:
                checklidar = True
             
        else:
            cost_map = occ#, _, _, _ = update_cost_map(bgr)
            

            fow = 100
            
               

        #cv2.imshow("Obstacle map", obstacle_map)
        
        grid_size = occ.shape[0]       # 400
        robot_px = grid_size // 2      # 200
        robot_py = grid_size // 2      # 200
        
        
        goal_forward_dist = 150
        goal_roi_width = 200
        goal_roi_height = 100
        
        # Initialize navigation state
        goal = None
        navigating = False
        
        # 1. Check if waypoint navigation is active
        if navigator.active:
            print("in active")
            # Convert current GPS to pixel coordinates
            current_pixel = gps_to_pixel(latitude, longitude, navigator)  
            
            # Update position and get next waypoint goal
            navigating = navigator.update_position(current_pixel)
            goal = navigator.get_goal_point()
        
        # 2. Fallback to lane navigation when waypoints are complete
        if not goal and not navigator.active:
            if navigator.check_near_first_waypoint((x, y), threshold=200):
                # Reset waypoint navigation if near start
                navigator.current_idx = 0
                navigator.path_completed = False
                navigator.isactive = True
            else:
                # Use lane-based navigation
                goal = get_next_goal_lane(
                    robot_xy=(x, y),
                    robot_heading=heading,
                    lane_mask=lane_mask_raw_valid,
                    obstacle_map=obstacle_map,
                    drivable_mask=drivable_area,
                    forward_dist=150,
                    roi_width=200,
                    roi_height=160
                )
        
        

        #4. if not goal & check lidar
        if checklidar and not goal:
            # 1. Find safe retreat direction using occupancy map
            vec_goal = get_safe_retreat_vec(occ, drivable_area, lane_mask_raw_valid)

            goal = convert_to_world(vec_goal, grid_resolution=0.1)
            #if goal:
            #    goal = convert_to_world(goal)
            print(f"Emergency retreat to {goal}")
           
        # 3. Apply obstacle padding to ANY valid goal
        if goal:
            x_goal, y_goal = pad_goal_from_obstacles(
                goal[0], goal[1], 
                obstacle_map, 
                pad_px=30
            )
            x_goal, y_goal = pad_goal_from_lines(x_goal, y_goal, lines_valid, pad_px=30)
            goal = (x_goal, y_goal)
        print(f"goal {goal} after padding")
        path = node.plan_rrt(drivable_area, (x, y), goal) if goal else None
        if path:
            pathgps = gpsconverter.convert_path(path)
            print(f"Path: {path}")
            pathimg =  bgr.copy()
            h, w = bgr.shape[:2]
            converted = [(int(x), int(h - y)) for x, y in path]

            pathimg = bgr.copy()

            for i in range(len(path) - 1):
                cv2.line(pathimg, tuple(map(int, path[i])), tuple(map(int, path[i + 1])), (0, 255, 0), 2)
            # (self, left_motor, right_motor, localize, img_h, img_w)

            cv2.imshow("Tru's Path", pathimg)
            controller = WebotsRosController(left, right,  gps=gps, imu=imu, origin_lat=None, origin_lon=None, localize=localizer, img_h=h, img_w= w)
            controller.path_set(path)
            controller.update()
            
            print("after update motors")
            node.publish_rrt_path(path)


        if cv2.waitKey(1) == ord('q'):
            break

    node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

