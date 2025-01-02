# {Crowd Density Estimation with YOLOv8}

=====================================

## Overview

---

This project is an object tracking and counting system that uses a YOLOv8 model to detect people in a video stream and track their movement. The system also counts the number of people moving up and down in a specific area.

## Features

---

- Object detection using YOLOv8 model
- Object tracking using a custom tracker class
- Counting of people moving up and down in a specific area
- Real-time video processing and display

## Requirements

---

- Python 3.7
- OpenCV 4.5
- Ultralytics YOLOv8 model
- Pandas library
- CVZone library

## Installation

---

1. Install the required libraries by running `pip install -r requirements.txt`
2. Download the YOLOv8 model from the Ultralytics website and place it in the project directory
3. Run the `main.py` file to start the object tracking and counting system

## Usage

---

1. Run the `main.py` file to start the object tracking and counting system
2. The system will display a video stream with detected objects and their tracking information
3. The system will also display the count of people moving up and down in a specific area

## Code Structure

---

- `tracker.py`: Custom tracker class for object tracking
- `main.py`: Main file for object detection, tracking, and counting
- `coco.txt`: Class names for YOLOv8 model

## Images

---

### Running Project

![Running Project](/src/demo.png)

This image shows the running project with object detection and tracking.

## Notes

---

- The system uses a confidence threshold of 0.2 for object detection
- The system uses a offset of 6 pixels for tracking objects
- The system counts people moving up and down in a specific area defined by two horizontal lines
