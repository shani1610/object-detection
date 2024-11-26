# object-detection


# VideoTools Setup Guide

## Setting up the Python Environment

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

# first pipeline: 
# Object Tracking Pipeline

This project demonstrates a simple object tracking pipeline using classical computer vision techniques. Below is a concise outline of the pipeline:

1. **Extract Frames**
   - Extract frames from the video to process sequentially.

2. **Compute Frame Difference**
   - Calculate the difference between the first two frames to detect motion.

3. **Binary Mask Using Thresholds**
   - Compute thresholds: 
     - `lower_threshold = mean - 2 * std`
     - `upper_threshold = mean + 2 * std`
   - Generate a binary mask by thresholding the difference image.

4. **Morphological Operations**
   - Apply erosion with a smaller kernel to minimize shrinking.
   - Apply dilation with a larger kernel to create a single contour line.

5. **Find Contours and Bounding Box**
   - Use `cv2.findContours` to detect contours.
   - Extract the bounding rectangle of the largest contour.

6. **Detect Features in Bounding Box**
   - Use the Shi-Tomasi Corner Detector to find good features within the bounding box.

7. **Track Features Across Frames**
   - Use Lucas-Kanade Optical Flow to track the detected features in subsequent frames.
   - Filter only the successfully tracked points (`st == 1`).

8. **Repeat for All Frames**
   - For each frame, update the features and continue tracking based on the previous frame.

# scripts\detect_jeep
![Description](output\tracked_video.gif)

