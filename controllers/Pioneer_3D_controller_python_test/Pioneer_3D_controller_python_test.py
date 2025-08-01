from ultralytics import YOLO
import cv2
import numpy as np
from controller import Robot, Keyboard

robot = Robot()
timestep = int(robot.getBasicTimeStep())


camera = robot.getDevice("camera")
camera.enable(timestep)
width, height = camera.getWidth(), camera.getHeight()

imu = robot.getDevice("IMU")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

model = YOLO(r"")

SPEED = 9.0

print("Drive with arrows; press 'q' in the window to quit.")

while robot.step(timestep) != -1:
  
    key = keyboard.getKey()
    if   key == Keyboard.UP:    ls = rs =  SPEED
    elif key == Keyboard.DOWN:  ls = rs = -SPEED
    elif key == Keyboard.LEFT:  ls, rs = -SPEED, SPEED
    elif key == Keyboard.RIGHT: ls, rs =  SPEED, -SPEED
    else:                       ls = rs = 0.0
    left_motor.setVelocity(ls)
    right_motor.setVelocity(rs)

    # --- camera + YOLO --- #
    buf = camera.getImage()
    if buf is None:
        continue
    frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
    img   = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results   = model.predict(img, conf=0.25)
    annotated = results[0].plot()

    
    ranges  = lidar.getRangeImage()
    fov     = lidar.getFov()
    max_r   = lidar.getMaxRange()
    num_pts = len(ranges)
    cx, cy  = width//2, height

    for i, r in enumerate(ranges):
        if not np.isfinite(r):
            continue
        r_clamped = min(r, max_r)
        angle = -fov/2 + i * (fov / (num_pts - 1))
        x = int(cx + np.sin(angle) * (r_clamped/max_r) * (width/2))
        y = int(cy - np.cos(angle) * (r_clamped/max_r) * (height/2))
        cv2.circle(annotated, (x, y), 2, (0,255,0), -1)


    rpy = imu.getRollPitchYaw()   
    g   = gps.getValues()         


    cv2.putText(annotated,
                f"IMU RPY: {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(annotated,
                f"GPS: X={g[0]:.3f}, Y={g[1]:.3f}, Z={g[2]:.3f}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

 
    print(f"IMU    (roll, pitch, yaw): {rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}")
    print(f"GPS    (x, y, z):            {g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}")
    print(f"LiDAR  ({num_pts} points):    {[f'{d:.4f}' for d in ranges]}")
    print("-" * 60)


    cv2.imshow("YOLO + LiDAR + IMU + GPS", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
