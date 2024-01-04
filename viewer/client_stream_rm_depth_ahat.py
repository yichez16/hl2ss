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
mode = hl2ss.StreamMode.MODE_1

# Framerate denominator (must be > 0)
divisor = 1 

# Depth encoding profile
profile_z = hl2ss.DepthProfile.SAME

# Video encoding profile
profile_ab = hl2ss.VideoProfile.H265_MAIN

# Image Saving Settings
image_counter = 0
save_path = './data/depth/'  # Update this to your desired save path

#------------------------------------------------------------------------------

if (mode == hl2ss.StreamMode.MODE_2):
    # ... [existing calibration code]
    quit()

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT, mode=mode, divisor=divisor, profile_z=profile_z, profile_ab=profile_ab)
client.open()

while (enable):
    data = client.get_next_packet()
    depth_scaled = (data.payload.depth / np.max(data.payload.depth) * 255).astype(np.uint8)
    ab_scaled = (data.payload.ab / np.max(data.payload.ab) * 255).astype(np.uint8)

    # Save depth and AB images
    cv2.imwrite(f'{save_path}depth_image_{image_counter}.png', depth_scaled)
    # cv2.imwrite(f'{save_path}ab_image_{image_counter}.png', ab_scaled)
    image_counter += 1

    # Display the images
    cv2.imshow('Depth', depth_scaled)
    cv2.imshow('AB', ab_scaled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
client.close()
listener.join()
cv2.destroyAllWindows()
