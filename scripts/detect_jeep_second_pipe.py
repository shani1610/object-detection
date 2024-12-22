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
 
if show_graphs:
    plt.imshow(img3),plt.show()

# Extract locations of matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate the homography matrix and filter outliers
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # src for frame1 keypoints, dst for frame2 keypoints
matches_mask = mask.ravel().tolist()

# Draw inliers
draw_params = dict(matchColor=(0, 255, 0),  # Green matches
                   singlePointColor=None,
                   matchesMask=matches_mask,  # Draw only inliers
                   flags=cv2.DrawMatchesFlags_DEFAULT)

img4 = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, **draw_params)
 
if show_graphs:
    # Display the matches
    plt.figure(figsize=(15, 10))
    plt.imshow(img4)
    plt.title("Matches After RANSAC")
    plt.show()

# Warp frame2 to align with frame1
height, width = frame1.shape
aligned_frame2 = cv2.warpPerspective(frame2, H, (width, height))

if show_graphs:
    # Visualize the aligned result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(frame1, cmap='gray'), plt.title("Frame 1 (Reference)")
    plt.subplot(1, 2, 2), plt.imshow(aligned_frame2, cmap='gray'), plt.title("Aligned Frame 2")
    plt.show()


# Extract inliers and outliers from RANSAC
inliers = np.where(mask.ravel() == 1)[0]
outliers = np.where(mask.ravel() == 0)[0]

# Separate inlier and outlier points
inlier_src_pts = src_pts[inliers] # src for frame1 keypoints, dst for frame2 keypoints
inlier_dst_pts = dst_pts[inliers]

outlier_src_pts = src_pts[outliers]
outlier_dst_pts = dst_pts[outliers]

# Visualize outliers (moving object points) on frame1
frame1_outliers = frame1.copy()
for pt in outlier_src_pts: # outlier_src_pts in frame1
    x, y = int(pt[0][0]), int(pt[0][1])
    cv2.circle(frame1_outliers, (x, y), 5, (255, 0, 0), -1)  # Red circles for outliers

if show_graphs:
    # Visualize the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(frame1, cmap='gray'), plt.title("Original Frame 1")
    plt.subplot(1, 2, 2), plt.imshow(frame1_outliers, cmap='gray'), plt.title("Detected Outliers (Moving Object)")
    plt.show()

if show_graphs:
    plt.figure(figsize=(10, 5))
    plt.scatter([pt[0][0] for pt in outlier_src_pts], [pt[0][1] for pt in outlier_src_pts], color='blue')
    plt.title("Outlier Points Before Clustering")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

k = 2

# clustering that includes the intensity
z = []
for pt in outlier_src_pts: # outlier_src_pts in frame1
    x, y = int(pt[0][0]), int(pt[0][1]) # we can cluster based on spatial proximity.
    intensity = frame1[y][x]  # Add intensity value
    z.append([x, y, intensity])  # Include intensity as a third feature

# convert to np.float32
Z = np.float32(z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

if show_graphs:
    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.title(f"K-Means Clustering, k = {k}, spatial proximity and intensity")
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()

# clustering that doesn't includes the intensity, better for this
z = []
for pt in outlier_src_pts: # outlier_src_pts in frame1
    x, y = int(pt[0][0]), int(pt[0][1]) # we can cluster based on spatial proximity.
    z.append([x, y])

# convert to np.float32
Z = np.float32(z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # when to stop, num of iterations, epsilon (the desired accuracy)
ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

if show_graphs:
    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.title(f"K-Means Clustering, k = {k}, spatial proximity only")
    plt.show()

print("Cluster 0 points:", A)
print("Cluster 1 points:", B)

# lets track only the biggest cluster points
# Identify the cluster with the most points
object_cluster = A if A.shape[0] > B.shape[0] else B

# Find the bounding box for the selected cluster
# Compute distances from the cluster centroid
distances = np.linalg.norm(object_cluster - np.mean(object_cluster, axis=0), axis=1)
threshold = np.percentile(distances, 90)  # Adjust this threshold as needed
filtered_points = object_cluster[distances <= threshold]

# Calculate the bounding box using filtered points
x_min, y_min = np.min(filtered_points, axis=0)
x_max, y_max = np.max(filtered_points, axis=0)
bounding_box = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
x, y, w, h = bounding_box
print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
if x < 0 or y < 0 or x+w > frame1.shape[1] or y+h > frame1.shape[0]:
    print("Error: Bounding box is out of frame bounds")
    
frame1_debug = frame1.copy()
cv2.rectangle(frame1_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box
if show_graphs:
    plt.imshow(frame1_debug, cmap='gray')
    plt.title("Bounding Box on Frame 1")
    plt.show()

# Update the initial location of the window for MeanShift
track_window = (x, y, w, h)

# Set up the ROI for tracking
# Visualize the ROI in the original frame
roi = frame1[y:y+h, x:x+w]

if show_graphs:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(roi, cmap='gray')
    plt.title("ROI (Grayscale)")
    plt.axis("off")

# Convert ROI to HSV
frame1_bgr = cv2.imread(frame1_path)
frame1_hsv = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2HSV)
roi_hsv = frame1_hsv[y:y+h, x:x+w]

if show_graphs:
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2RGB))
    plt.title("ROI in HSV")
    plt.axis("off")

# Visualize the mask applied to the HSV ROI
mask = cv2.inRange(roi_hsv, np.array((0., 50., 50.)), np.array((180., 255., 255.)))

if show_graphs:
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("ROI Mask")
    plt.axis("off")
    plt.show()

# Calculate and plot the ROI histogram
roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])

if show_graphs:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(roi_hist, color='blue')
    plt.title("Raw ROI Histogram (Hue)")
    plt.xlabel("Hue Bin")
    plt.ylabel("Frequency")

# Normalize the histogram and visualize
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

if show_graphs:
    plt.subplot(1, 2, 2)
    plt.plot(roi_hist, color='green')
    plt.title("Normalized ROI Histogram")
    plt.xlabel("Hue Bin")
    plt.ylabel("Frequency")
    plt.show()

# Setup the termination criteria for MeanShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# Initialize video writer
output_video_path = os.path.join(processed_data_folder, "tracked_video.avi")
frame_width = frame1.shape[1]
frame_height = frame1.shape[0]
fps = 10  # Adjust based on your video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Initialize video writer for the back-projection (dst) frames
# Initialize video writer for the back-projection (dst) frames (grayscale)
output_dst_path = os.path.join(processed_data_folder, "dst_video.avi")
out_dst = cv2.VideoWriter(output_dst_path, fourcc, fps, (frame_width, frame_height), 0)  # 0 indicates grayscale


# Iterate through all frames in sorted order
frame_paths = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".jpg")])

# Process each frame
for frame_path in frame_paths[1:]:
    curr_frame = cv2.imread(frame_path)
    hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    cv2.imshow("Back-Projection", dst)
    # Debug: Save dst to video
    dst_colorized = cv2.applyColorMap(cv2.convertScaleAbs(dst, alpha=255.0/dst.max()), cv2.COLORMAP_JET)  # Optional for visualization
    # Debug: Visualize and save the back-projection
    cv2.imshow("Back-Projection", dst)
    out_dst.write(dst)  # Write the grayscale back-projection directly

    # Apply MeanShift
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    print(f"Updated track_window: {track_window}")

    # Draw the rectangle
    x, y, w, h = track_window
    tracked_frame = cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Optional: Visualize the tracking result
    if show_graphs:
        cv2.imshow("Tracking", tracked_frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit early
            break
    # Write the frame to the output video
    out.write(tracked_frame)

# Release resources
out.release()
out_dst.release()
cv2.destroyAllWindows()
print(f"Output video saved at {output_video_path}")

# convert avi to gif
video_tools = VideoTools()
output_video_gif_path = os.path.join(processed_data_folder, "tracked_video_gif.gif")
output_dst_gif_path = os.path.join(processed_data_folder, "dst_video_gif.gif")
video_tools.avi2gif(output_video_path, output_video_gif_path)
video_tools.avi2gif(output_dst_path, output_dst_gif_path)