#!/bin/bash
# Tải module v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1

# Đẩy luồng từ camera thật vào thiết bị ảo
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! video/x-raw, width=640, height=480, framerate=30/1 ! v4l2sink device=/dev/video10