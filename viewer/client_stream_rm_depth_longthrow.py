#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in long throw mode
# and plays it. The resolution is 320x288 @ 5 FPS. The stream supports three
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop. 
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.0.110"

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Framerate denominator (must be > 0)
# Effective framerate is framerate / divisor
divisor = 1

#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    data = hl2ss_lnm.download_calibration_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    print('Calibration data')
    print('Image point to unit plane')
    print(data.uv2xy)
    print('Extrinsics')
    print(data.extrinsics)
    print(f'Scale: {data.scale}')
    print('Undistort map')
    print(data.undistort_map)
    print('Intrinsics (undistorted only)')
    print(data.intrinsics)
    quit()

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, mode=mode, divisor=divisor)
client.open()

# Initialize VideoWriters
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_depth = cv2.VideoWriter('./data/depth_long.avi', fourcc, 5, (320, 288), isColor=False)
out_ab = cv2.VideoWriter('./data/ab_long.avi', fourcc, 5, (320, 288), isColor=False)

while (enable):
    data = client.get_next_packet()

    print(f'Pose at time {data.timestamp}')
    print(data.pose)

    # Scale and write the frames
    depth_scaled = (data.payload.depth / np.max(data.payload.depth) * 255).astype(np.uint8)
    ab_scaled = (data.payload.ab / np.max(data.payload.ab) * 255).astype(np.uint8)

    
    cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
    cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility

    out_depth.write(depth_scaled)
    out_ab.write(ab_scaled)
    cv2.waitKey(1)

# Release the VideoWriters and other resources
out_depth.release()
out_ab.release()
client.close()
listener.join()


