# Object Detection
This project demonstrates a simple object tracking pipeline using classical computer vision techniques. Below is a concise outline of the pipeline:
I did it for self study and expiriments.

# Table of Contents

- [Setup](#setup)
- [Pipelines](#pipelines)
  - [Pipeline1](##pipeline1)
  - [Pipeline2](##pipeline2)
- [Details](#Details)

  
# Setup

1. **Create a new virtual environment**  
   Run the following command to create a virtual environment named `videotools_env`:
   ```bash
   python3 -m venv cvenv

2. **Activate the virtual environment**
On Linux or macOS:
```bash
source cvenv/bin/activate
```
On Windows:
```bash
.\cvenv\Scripts\activate
```
Install required dependencies
Use the provided requirements.txt file to install all necessary libraries:

```bash
pip install -r requirements.txt
```

run 
```bash 
python VideoTools.py extract_frames "path/to/video.mp4" "path/to/output" --num_frames 50
```

# Pipelines

##  Pipeline1

1. **Extract Frames**
2. **Compute Frame Difference**
3. **Binary Mask Using Thresholds**
4. **Morphological Operations**
5. **Find Contours and Bounding Box**
6. **Detect Features in Bounding Box**
7. **Track Features Across Frames**
8. **Repeat for All Frames**


## Pipeline2

1. **Extract Frames**
2. **Keypoint Detection with SIFT**
3. **Feature Matching**
4. **Cluster Outlier Points**
5. **Bounding Box for the Main Cluster**
6. **ROI Histogram Creation**
7. **Tracking with MeanShift**

# Details 

a detailed report can be found [here](https://github.com/shani1610/object-detection/blob/main/report/README.md)

