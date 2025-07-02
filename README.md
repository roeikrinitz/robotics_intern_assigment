# Situational Awareness PoC – Robotics Intern Task

## Project Summary

This ROS 2 node fuses a 128-beam LiDAR and an Intel RealSense RGB camera to generate a minimal world map (2D occupancy grid). The system runs at 5 FPS and leverages a pretrained neural network (Mask R-CNN) to detect obstacles from the camera, combining them with precise geometry from LiDAR for real-time environmental awareness.

---

## Features

- ✅ Real-time fusion of LiDAR and RGB camera
- ✅ Pretrained Mask R-CNN model for object detection
- ✅ Generation of 2D occupancy grid map (fused map)
- ✅ ROS 2-compatible (`rclpy`, `sensor_msgs`, `nav_msgs`)
- ✅ Runs at 5 frames per second

---

##  Requirements

### Python (via pip)
Install using:
pip install -r requirements.txt


### Ros 2
Coming soon...