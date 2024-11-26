import sys
import os

# Get the project directory and append the tools directory to sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, 'tools'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from VideoTools import VideoTools


show_graphs = False

# Paths
data_folder = "data"
processed_data_folder = "processed_data"
gif_file = os.path.join(data_folder, "jeep.gif")
mp4_file = os.path.join(processed_data_folder, "jeep.mp4")
frames_folder = os.path.join(processed_data_folder, "jeep")

first_extraction = False
if first_extraction:
    # Ensure processed_data folder exists
    os.makedirs(processed_data_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    # Step 1: Convert GIF to MP4
    video_tools = VideoTools()
    video_tools.gif2mp4(gif_file, mp4_file)

    # Step 2: Extract 10 frames
    video_tools.extract_frames(mp4_file, frames_folder, num_frames=10)

# Step 3: Compute difference between two frames
frame1_path = os.path.join(frames_folder, "0000.jpg")
frame2_path = os.path.join(frames_folder, "0001.jpg")

# Load two grayscale frames
frame1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(frame1,None)
kp2, des2 = sift.detectAndCompute(frame2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
# Apply ratio test
good_matches = []
good_matches_sublist = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches_sublist.append([m])
        good_matches.append(m)

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,good_matches_sublist,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
plt.imshow(img3),plt.show()

# Extract locations of matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate the homography matrix and filter outliers
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# Draw inliers
draw_params = dict(matchColor=(0, 255, 0),  # Green matches
                   singlePointColor=None,
                   matchesMask=matches_mask,  # Draw only inliers
                   flags=cv2.DrawMatchesFlags_DEFAULT)

img4 = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, **draw_params)

# Display the matches
plt.figure(figsize=(15, 10))
plt.imshow(img4)
plt.title("Matches After RANSAC")
plt.show()

# Warp frame2 to align with frame1
height, width = frame1.shape
aligned_frame2 = cv2.warpPerspective(frame2, H, (width, height))

# Visualize the aligned result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(frame1, cmap='gray'), plt.title("Frame 1 (Reference)")
plt.subplot(1, 2, 2), plt.imshow(aligned_frame2, cmap='gray'), plt.title("Aligned Frame 2")
plt.show()


# Extract inliers and outliers from RANSAC
inliers = np.where(mask.ravel() == 1)[0]
outliers = np.where(mask.ravel() == 0)[0]

# Separate inlier and outlier points
inlier_src_pts = src_pts[inliers]
inlier_dst_pts = dst_pts[inliers]

outlier_src_pts = src_pts[outliers]
outlier_dst_pts = dst_pts[outliers]

# Visualize outliers (moving object points) on frame1
frame1_outliers = frame1.copy()
for pt in outlier_src_pts:
    x, y = int(pt[0][0]), int(pt[0][1])
    cv2.circle(frame1_outliers, (x, y), 5, (255, 0, 0), -1)  # Red circles for outliers

# Visualize the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(frame1, cmap='gray'), plt.title("Original Frame 1")
plt.subplot(1, 2, 2), plt.imshow(frame1_outliers, cmap='gray'), plt.title("Detected Outliers (Moving Object)")
plt.show()
