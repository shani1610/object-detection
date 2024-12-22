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


1. **Extract Frames**
Extracted individual frames from the input video (jeep.gif) and saved them as images (.jpg format). This was done to process frames sequentially in later steps.

2. **Keypoint Detection with SIFT:**

   - Detected keypoints in the first two frames (frame1 and frame2) using the SIFT algorithm.
   - Keypoints are distinctive points in the image, useful for tracking and alignment.

 3. **Feature Matching:**

   - Matched the detected keypoints between the two frames using Brute-Force Matcher (BFMatcher).
   - Applied a ratio test to filter matches, retaining only those where the closest match was significantly better than the second closest.
   - Homography Estimation with RANSAC
   - Extracted the matched keypoints into source (src_pts) and destination (dst_pts) arrays.
   - Used RANSAC to estimate a homography matrix that maps points in frame1 to frame2:
   - RANSAC robustly filtered outliers by iteratively fitting models to random subsets of points and choosing the best model.
   - Visualized the inliers and outliers on frame1 to verify the moving object regions.

 4. **Cluster Outlier Points**
 
   - Clustered the outliers (likely belonging to the moving object) into two groups using K-Means:
   - Clustering was performed both with and without intensity as a feature.
   - Selected the largest cluster to focus on the main moving object.

5. **Bounding Box for the Main Cluster**
   
   - Calculated the bounding box around the largest cluster of outlier points:
   - Used the mean and distances of the cluster points to filter stray points and tighten the bounding box.

7. **ROI Histogram Creation**
   
   - Extracted the Region of Interest (ROI) corresponding to the bounding box.
   - Converted the ROI from grayscale to the HSV color space.
   - Calculated and normalized the hue channel histogram of the ROI:
   - Normalization ensured robustness against illumination changes.

7. **Tracking with MeanShift**
   - Used the histogram from the previous step to create a back-projection map for each frame:
   - Bright regions in the map correspond to areas similar to the object's histogram.
   - Applied MeanShift tracking to locate the bounding box of the object in subsequent frames.
   - Saved both the tracking output and the back-projection map as videos for evaluation.

