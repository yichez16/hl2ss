from pynput import keyboard
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import os
import time

# Wait for 3 seconds
time.sleep(3)


# Settings --------------------------------------------------------------------
host = "192.168.0.110"
mode = hl2ss.StreamMode.MODE_1  # Mode selection
divisor = 1
enable_mrc = False
# RGB Camera Settings
rgb_width = 1920
rgb_height = 1080
rgb_framerate = 30
rgb_profile = hl2ss.VideoProfile.H265_MAIN
rgb_decoded_format = 'bgr24'
# Depth Camera Settings
depth_profile_z = hl2ss.DepthProfile.SAME
depth_profile_ab = hl2ss.VideoProfile.H265_MAIN

# Image Saving Settings
image_counter = 0
save_path_rgb = './data_5_false/rgb/'
save_path_depth = './data_5_false/depth/'


# Create directories if they don't exist
os.makedirs(save_path_rgb, exist_ok=True)
os.makedirs(save_path_depth, exist_ok=True)

# Initialize Keyboard Listener
enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Initialize Clients for RGB and Depth

# hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc)

rgb_client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=rgb_width, height=rgb_height, framerate=rgb_framerate, divisor=divisor, profile=rgb_profile, decoded_format=rgb_decoded_format)
depth_client = hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, mode=mode, divisor=divisor, profile_z=depth_profile_z, profile_ab=depth_profile_ab)

rgb_client.open()
depth_client.open()

# Main Loop
while enable:
    # # Retrieve RGB data
    rgb_data = rgb_client.get_next_packet()
    cv2.imshow('RGB Video', rgb_data.payload.image)
    rgb_frame_filename = f"{save_path_rgb}frame_{image_counter}.png"
    cv2.imwrite(rgb_frame_filename, rgb_data.payload.image)

    # Retrieve Depth data
    depth_data = depth_client.get_next_packet()
    depth_scaled = (depth_data.payload.depth / np.max(depth_data.payload.depth) * 255).astype(np.uint8)
    cv2.imshow('Depth', depth_scaled)
    depth_frame_filename = f'{save_path_depth}depth_image_{image_counter}.png'
    cv2.imwrite(depth_frame_filename, depth_scaled)

    image_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
# rgb_client.close()
depth_client.close()
listener.join()
cv2.destroyAllWindows()
