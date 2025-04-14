# Inventory Object Detection and Classification

## Overview
This project implements a system for detecting and classifying objects in warehouse environments using computer vision techniques. The primary focus is on visual odometry to track camera position, which is then integrated with object detection to determine the relative positions of objects in the scene.

## Problem Statement
Developing a system that detects, classifies, and localizes products/objects in a warehouse environment using YOLO model and visual odometry, with applications in automated inventory management for industrial settings.

## Key Features
- Camera trajectory and position estimation through visual odometry
- Object detection and classification using YOLOv5
- Object classification based on shape, markings, and color
- Relative positioning of detected objects in the scene
- Comprehensive report generation of object positions

## Input/Output
- **Input**: Video footage captured from a camera moving through a "warehouse" environment
- **Output**: 
  - Camera position and trajectory data
  - Classification of detected objects
  - Relative positions of objects in the environment
