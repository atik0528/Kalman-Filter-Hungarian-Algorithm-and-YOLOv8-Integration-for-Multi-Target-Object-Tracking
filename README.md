# Kalman Filter, Hungarian Algorithm and YOLOv8 Integration for Multi-Target Tracking

An advanced approach for efficient multi-target tracking combining Kalman Filtering for state estimation, the Hungarian algorithm for optimal assignment, and YOLOv8 for object detection. Perfect for real-time tracking in applications like autonomous vehicles, robotics, and video surveillance.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In real-world scenarios, accurately tracking multiple moving objects is a challenging task, particularly in dynamic environments with occlusions and measurement noise. This project addresses these challenges by combining three powerful tools: YOLOv8 for detecting objects in real-time, the Kalman Filter for estimating the state of a moving object over time, and the Hungarian Algorithm for solving the optimal assignment problem.

YOLOv8 provides real-time object detection capabilities, which, when combined with the Kalman Filter and Hungarian Algorithm, results in a comprehensive solution for multi-target tracking in dynamic environments.

## Features
- **Kalman Filter**: Predicts the state of dynamic systems and corrects estimations based on measurements.
- **Hungarian Algorithm**: Finds optimal matches between predicted object positions and detected measurements.
- **YOLOv8 Integration**: Utilizes state-of-the-art object detection to identify targets in each frame.
- **Multi-Target Tracking**: Supports tracking multiple objects concurrently.
- **Real-Time Processing**: Designed for efficiency and can handle real-time tracking requirements.
- **Scalable**: Applicable to various domains including computer vision, robotics, and autonomous systems.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Atik0528/Kalman Filter, Hungarian Algorithm and YOLOv8 Integration for Multi-Target Tracking.git
cd repo
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.8
- Numpy
- OpenCV (optional, for visualizing tracking results)
- Scipy
- PyTorch (for YOLOv8 integration)
- Ultralytics (for YOLOv8 model)

These dependencies can be installed using the provided `requirements.txt` file.

## Usage
After installing the necessary packages, you can run the Jupyter Notebook to see the implementation in action:

```bash
jupyter notebook Implementation_kalman_Hungarian.ipynb
```

To run the YOLOv8 integration script for real-time detection and tracking:

```bash
python Implementation_kalman_Hungarian_Yolo_V8.py
```

In the notebook and scripts, you'll find step-by-step explanations on the implementation of YOLOv8, the Kalman Filter, and the Hungarian Algorithm. You can modify the parameters as needed to suit your specific use case.

## Implementation Details
### YOLOv8
**YOLOv8** is the latest version of the "You Only Look Once" (YOLO) family of object detection models. It offers high-speed and accurate detection, making it suitable for real-time applications. In this implementation, YOLOv8 is used to detect multiple objects in each frame, providing the input for the Kalman Filter and Hungarian Algorithm.

### Kalman Filter
The Kalman Filter is an optimal estimation algorithm that is commonly used to estimate the state of a linear dynamic system. It consists of two primary steps:

1. **Prediction Step**: Predict the next state of the system and the associated uncertainty.
2. **Update Step**: Correct the predicted state based on new measurements.

The Kalman Filter works effectively in environments with Gaussian noise, which makes it perfect for state estimation in real-world applications.

In this implementation, the Kalman Filter is used to track the position and velocity of objects over time. The state vector includes both the position and velocity, which allows the filter to account for the movement dynamics of the tracked objects.

### Hungarian Algorithm
The **Hungarian Algorithm**, also known as the Kuhn-Munkres algorithm, is used to solve assignment problems where there are multiple agents and tasks, and the goal is to find the optimal assignment with minimum cost. In the context of tracking, it is used to associate the predicted object locations from the Kalman Filter with new measurements (detections) in each frame.

The Hungarian Algorithm operates in polynomial time and provides an optimal solution, making it suitable for scenarios with numerous targets and measurements.

## Results
The combination of YOLOv8, the Kalman Filter, and the Hungarian Algorithm has been extensively tested in a simulated environment with multiple moving objects. The algorithm is capable of:

- **Handling occlusions**: The Kalman Filter can predict the state of an object even when it is temporarily occluded.
- **Robust Matching**: The Hungarian Algorithm ensures that measurements are correctly assigned to their respective objects, minimizing misassignments.
- **Efficient State Estimation**: The Kalman Filter provides smooth and accurate position and velocity estimates of all tracked objects.
- **Accurate Detection**: YOLOv8 provides precise object detection, enhancing the overall tracking performance.

### Example Scenario
Below is a simple simulation with several targets moving across a 2D plane. The YOLOv8 model detects the objects, the Kalman Filter estimates their positions, and the Hungarian Algorithm assigns new measurements to these tracked positions.

## Examples
To illustrate the capabilities of this approach, an example is included in the Jupyter notebook where multiple objects move across a 2D environment. The notebook demonstrates how YOLOv8 detects objects, the Kalman Filter predicts the next state, and the Hungarian Algorithm assigns measurements to predicted states.

The notebook contains:
- Visualization of detected, predicted, and corrected states over time.
- An implementation of the cost matrix for assignment using the Hungarian Algorithm.
- Examples with different types of noise to demonstrate the robustness of the Kalman Filter.

## Contributing
Contributions are always welcome! Whether it's bug fixes, new features, or improvements to documentation, your input is highly valued.

To contribute:
1. Fork this repository.
2. Create a branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or suggestions regarding this project, please feel free to reach out via GitHub issues.

