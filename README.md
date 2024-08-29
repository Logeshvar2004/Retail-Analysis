# Retail-Analysis

## Real-Time Person Tracking with Age and Gender Classification

### Overview
This repository contains a Python application for real-time person tracking using YOLOv8. It integrates age and gender classification and tracks individuals' movements across video frames. The application utilizes Kalman Filters for tracking and recording data on persons standing in the same location for a specified period. Data is saved to a CSV file only if certain conditions are met, such as minimal movement and dwell time.

### Features
- **Real-Time Tracking:** Uses YOLOv8 to detect and track multiple persons in a video.
- **Age and Gender Classification:** Integrates pre-trained Caffe models for age and gender detection.
- **Kalman Filter:** Applies Kalman Filter for smooth tracking of object positions.
- **Dwell Time Calculation:** Records the dwell time of individuals standing in the same place for more than a specified duration.
- **Data Logging:** Saves relevant data (ID, age, gender, dwell time) to a CSV file if the person remains in the same position.

### Requirements
- Python 3.10 or later
- OpenCV
- PyTorch
- Ultralyics YOLO
- Caffe (for age and gender models)
- Pandas
- NumPy
- FilterPy

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/repository-name.git
    cd repository-name
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download YOLOv8 weights and place them in the appropriate directory.

4. Download the pre-trained Caffe models for age and gender classification and place them in the `Retail` directory.

### Usage
1. Modify the `video_path` variable in the script to point to your video file.
2. Run the script:
    ```bash
    python main.py
    ```
3. The application will process the video in real time, displaying detected persons and their details. Data will be saved to `output.csv` if the conditions are met.

### CSV File Structure
The output CSV file contains:
- `ID`: Unique identifier for each tracked person.
- `Gender`: Predicted gender of the person.
- `Age`: Predicted age range of the person.
- `Dwell Time`: The time the person has been detected standing in the same place.

### Acknowledgements
- YOLOv8 for object detection
- Caffe for age and gender classification models
- OpenCV and PyTorch for computer vision and deep learning functionalities
