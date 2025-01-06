# Motion Detection Methods

This repository contains implementations of two motion detection methods: Optical Flow and Frame Difference. These methods were inspired by the article "A Method to Improve the Data Screening Accuracy of Infrared Camera Trapping".

## Optical Flow Method

The Optical Flow method is used to detect motion by analyzing the apparent motion of objects between two consecutive frames. It is particularly useful for detecting small movements and can be applied in various computer vision tasks.

## Frame Difference Method

The Frame Difference method is a simpler approach to motion detection. It compares the difference between two consecutive frames to identify moving objects. This method is computationally efficient and is often used in real-time applications.

## Usage

To use these methods, simply clone the repository and run the provided scripts. Make sure you have the necessary dependencies installed, such as OpenCV.

```bash
git clone https://github.com/your-username/motion-detection-methods.git
cd motion-detection-methods
pip install opencv-python
