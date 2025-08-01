"""
    File: WaypointNavigator.py Calculate and navigate through waypoints using GPS coordinates.

    This script reads waypoints from a YAML file, converts GPS coordinates to UTM, 
    and then to pixel coordinates for navigation. It supports dynamic recalibration 
    based on observed landmarks and checks proximity to waypoints for navigation.

    Classes
        Waypoint: Represents a single waypoint with GPS, UTM, and pixel coordinates.
            Functions:
                __init__: Initializes a waypoint with GPS coordinates.
                __repr__: String representation of the waypoint.
        WaypointNavigator: Manages a list of waypoints, converts coordinates,
        and handles navigation logic.
            Functions:
                __init__: Initializes the navigator with waypoints from a YAML file.
                load_waypoints: Loads waypoints from a YAML file.
                convert_waypoints_to_utm: Converts GPS coordinates to UTM.
                convert_utm_to_pixels: Converts UTM coordinates to pixel coordinates.
                dynamic_recalibrate: Improves accuracy using a known landmark position.
                check_near_first_waypoint: Checks proximity to the first waypoint.
                get_current_target: Gets the current target waypoint in all coordinate systems.
                update_position: Updates the current position and checks if the waypoint is reached.
                camera_to_map_pixel: Converts camera coordinates to map pixel coordinates.
                dynamic_recalibrate_from_camera: Recalibrates using camera observations of known landmarks.
                get_goal_point: Gets the current waypoint goal in pixel coordinates.
"""

import yaml
import utm
import math

class Waypoint:
    """
        Represents a waypoint with GPS, UTM, and pixel coordinates.

        Variables:
            id (str): Unique identifier for the waypoint.
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.
            utm_easting (float): UTM easting in meters.
            utm_northing (float): UTM northing in meters.
            pixel_x (int): X coordinate in pixel
            pixel_y (int): Y coordinate in pixel
    """

    def __init__(self, wp_id, lat, lon):
        """
            Initialize a waypoint with GPS coordinates.

            Args:
                wp_id (str): Unique identifier for the waypoint.
              lat (float): Latitude in degrees.
              lon (float): Longitude in degrees.
        """
        self.id = wp_id
        self.lat = lat
        self.lon = lon
        self.utm_easting = None
        self.utm_northing = None
        self.pixel_x = None
        self.pixel_y = None
        self.zone_number = None
        self.zone_letter = None

    def __repr__(self):
        return (f"Waypoint {self.id}\n"
                f"  GPS: ({self.lat:.6f}, {self.lon:.6f})\n"
                f"  UTM: {self.utm_easting:.1f}E {self.utm_northing:.1f}N\n"
                f"  Pixel: ({self.pixel_x}, {self.pixel_y})")

class WaypointNavigator:
    """
        Manages navigation through a list of waypoints.

        Varibles:
            waypoints (list): List of Waypoint objects.
            current_idx (int): Index of the current waypoint.
            pixels_per_meter (int): Scale factor for converting UTM to pixel coordinates.
            path_completed (bool): Flag indicating if the path is completed.
            origin_easting (float): UTM easting of the origin.
            origin_northing (float): UTM northing of the origin.

    """

    def __init__(self, yaml_file, origin_lat, origin_lon, pixels_per_meter=10):
        """
        Initialize the navigator with waypoints from a YAML file and starting GPS position.

        Args:
            yaml_file: Path to waypoints YAML
            origin_lat: Starting GPS latitude
            origin_lon: Starting GPS longitude
            pixels_per_meter: Initial scale estimate
        """
        self.waypoints = []
        self.current_idx = 0
        self.pixels_per_meter = pixels_per_meter
        self.path_completed = False
        self.isactive = False
        # Add storage for the UTM origin and pixel origin
        self.utm_origin = None  # (easting, northing)
        self.pixel_origin = None  # (pixel_x, pixel_y)

        # Set initial UTM origin from starting position
        self.origin_easting, self.origin_northing, \
        self.zone_number, self.zone_letter = utm.from_latlon(origin_lat, origin_lon)
        
        # Load and convert waypoints
        self.load_waypoints(yaml_file)
        self.convert_waypoints_to_utm()
        self.convert_utm_to_pixels()

    def load_waypoints(self, yaml_file):
        """
            Load waypoints from a YAML file.
        """
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            
        for wp_data in data['waypoints']:
            wp = Waypoint(
                wp_data['id'],
                wp_data['lat'],
                wp_data['lon']
            )
            self.waypoints.append(wp)

    def convert_waypoints_to_utm(self):
        """
        
        Convert all waypoints to UTM using origin's zone
        
        This uses the UTM zone of the origin GPS coordinates.
        
        """
        for wp in self.waypoints:
            wp.utm_easting, wp.utm_northing, _, _ = utm.from_latlon(
                wp.lat, wp.lon,
                force_zone_number=self.zone_number,
                force_zone_letter=self.zone_letter
            )

    def convert_utm_to_pixels(self):
        """Convert UTM coordinates to pixel space relative to origin."""
        for wp in self.waypoints:
            dx = wp.utm_easting - self.origin_easting
            dy = wp.utm_northing - self.origin_northing
        
            # Use consistent naming (pixel_x/pixel_y)
            wp.pixel_x = int(dx * self.pixels_per_meter)
            wp.pixel_y = int(-dy * self.pixels_per_meter)  


    



    def dynamic_recalibrate(self, observed_pixel, known_gps):
        """
        Improve accuracy using a known landmark position
        Args:
            observed_pixel: (x,y) pixels where landmark appears
            known_gps: (lat,lon) of landmark
        """
        # Convert known GPS to UTM
        easting, northing, _, _ = utm.from_latlon(
            known_gps[0], known_gps[1],
            force_zone_number=self.zone_number,
            force_zone_letter=self.zone_letter
        )
        
        # Calculate ideal vs observed
        ideal_dx = easting - self.origin_easting
        ideal_dy = northing - self.origin_northing
        
        # Update scaling factors
        if ideal_dx != 0:
            self.pixels_per_meter = (self.pixels_per_meter + 
                                    observed_pixel[0]/ideal_dx) / 2
        if ideal_dy != 0:
            self.pixels_per_meter = (self.pixels_per_meter +
                                    abs(observed_pixel[1])/ideal_dy) / 2
        
        # Reconvert all waypoints with new scale
        self.convert_utm_to_pixels()

    def check_near_first_waypoint(self, current_pixel, threshold=200):
        """
        
        Check proximity to first waypoint (pixels)
        
        Args:
            current_pixel: (x,y) pixel position of the robot
            threshold: distance in pixels to consider "near"

        """
        first_wp = self.waypoints[0]
        distance = math.hypot(
            first_wp.pixel_x - current_pixel[0],
            first_wp.pixel_y - current_pixel[1]
        )
        return distance <= threshold

    def get_current_target(self):
        """
            Get the current target waypoint in all coordinate systems.
            Returns current target in all coordinate systems"""
        if self.path_completed:
            return None
            
        wp = self.waypoints[self.current_idx]
        return {
            'id': wp.id,
            'gps': (wp.lat, wp.lon),
            'utm': (wp.utm_easting, wp.utm_northing),
            'pixel': (wp.pixel_x, wp.pixel_y)
        }

    def update_position(self, current_pixel, threshold=200):
        """
        
        Progress to next waypoint if within threshold (pixels)
        
        Args:
            current_pixel: (x,y) pixel position of the robot
            threshold: distance in pixels to consider "reached"

        Returns:
            bool: True if the waypoint was reached, False otherwise.
        """
        if self.path_completed:
            return True
            
        target = self.get_current_target()
        if not target:
            return False
            
        distance = math.hypot(
            target['pixel'][0] - current_pixel[0],
            target['pixel'][1] - current_pixel[1]
        )
        
        if distance <= threshold:
            if self.current_idx == len(self.waypoints) - 1:
                self.path_completed = True
            else:
                self.current_idx += 1
            return True
            
        self.isactive = False
        return False

    def camera_to_map_pixel(self, camera_x, camera_y, robot_heading, current_robot_pixel):
        """
        Convert camera coordinates to navigator's map pixel coordinates.
        
        Args:
            camera_x, camera_y: Pixel coordinates in camera image (0-1280, 0-720)
            robot_heading: Current heading in radians
            current_robot_pixel: Robot's current position in map pixels (x, y)
            
        Returns:
            (x, y) in map pixel coordinates
        """
        # Camera parameters
        CAM_WIDTH = 1280
        CAM_HEIGHT = 720
        HFOV = 1.047  # 60 degrees in radians (typical Webots camera)
        
        # Calculate angle offset from center
        x_offset = camera_x - CAM_WIDTH/2
        angle = (x_offset / CAM_WIDTH) * HFOV
        
        # Estimate distance using vertical position (simple perspective approximation)
        ground_y = CAM_HEIGHT - camera_y  # Distance from bottom of image
        distance = (CAM_HEIGHT / (2 * math.tan(HFOV/2))) / ground_y  # In meters
        
        # Convert to map coordinates
        dx = distance * math.cos(robot_heading + angle) * self.pixels_per_meter
        dy = distance * math.sin(robot_heading + angle) * self.pixels_per_meter
        
        return (
            int(current_robot_pixel[0] + dx),
            int(current_robot_pixel[1] - dy) 
        )

    def dynamic_recalibrate_from_camera(self, camera_x, camera_y, 
                                      robot_heading, current_robot_pixel,
                                      known_gps):
        """
        Recalibrate using camera observations of known landmarks.
        
        Args:
            camera_x, camera_y: Pixel in camera image where landmark appears
            robot_heading: Current robot orientation (radians)
            current_robot_pixel: Robot's map position (x, y)
            known_gps: (lat, lon) of observed landmark
        """
        # Convert camera coordinates to map pixels
        observed_pixel = self.camera_to_map_pixel(
            camera_x, camera_y,
            robot_heading,
            current_robot_pixel
        )
        
        # Perform recalibration
        self.dynamic_recalibrate(observed_pixel, known_gps)


    def get_goal_point(self):
        """Gets the current waypoint goal in pixel coordinates."""
        if self.current_idx < len(self.waypoints):  
            wp = self.waypoints[self.current_idx]  
            return (wp.pixel_x, wp.pixel_y)         
        return None

    @property
    def active(self):
        """Check if navigation is still active."""
        return self.isactive
