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

# Compute the difference image
difference = cv2.absdiff(frame1, frame2)

# Calculate the histogram
hist = cv2.calcHist([difference], [0], None, [256], [0, 256])
hist_img = cv2.calcHist([frame1], [0], None, [256], [0, 256])

# Calculate mean and standard deviation
mean = np.mean(difference)
std = np.std(difference)

if show_graphs:
    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram of Difference Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram of Original Grayscale Image (Frame 1)")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist_img)
    plt.xlim([0, 256])
    plt.show()

# Set threshold based on 2 sigmas (95% rule)
lower_threshold = mean - 2 * std
upper_threshold = mean + 2 * std

# Apply threshold to create a binary mask
binary_mask = ((difference < lower_threshold) | (difference > upper_threshold)).astype(np.uint8) * 255

lower_threshold = mean - 1 * std
upper_threshold = mean + 1 * std
binary_mask_1 = ((difference < lower_threshold) | (difference > upper_threshold)).astype(np.uint8) * 255

lower_threshold = mean - 3 * std
upper_threshold = mean + 3 * std
binary_mask_3 = ((difference < lower_threshold) | (difference > upper_threshold)).astype(np.uint8) * 255

if show_graphs:
    # Display the results
    plt.figure(figsize=(20, 5))
    plt.subplot(3, 3, 1), plt.title("Frame 1"), plt.imshow(frame1, cmap='gray')
    plt.subplot(3, 3, 2), plt.title("Frame 2"), plt.imshow(frame2, cmap='gray')
    plt.subplot(3, 3, 3), plt.title("Differance with no processing"), plt.imshow(difference, cmap='gray')
    plt.subplot(3, 3, 4), plt.title("Thresholded Changes, 3*std"), plt.imshow(binary_mask_3, cmap='gray')
    plt.subplot(3, 3, 5), plt.title("Thresholded Changes, 2*std"), plt.imshow(binary_mask, cmap='gray')
    plt.subplot(3, 3, 6), plt.title("Thresholded Changes, 1*std"), plt.imshow(binary_mask_1, cmap='gray')
    plt.show()



# Use Morphological Operations to clean noise and fill gaps
kernel = np.ones((3,3), np.uint8) # used smaller kernel for the erosion for minimal shrinking of the foreground
erosion = cv2.erode(binary_mask, kernel, iterations = 1) 
kernel = np.ones((9,9), np.uint8) # used bigger kernel in order to get one contour line
dilation = cv2.dilate(binary_mask, kernel, iterations = 1)
noise_removed = cv2.dilate(erosion, kernel, iterations = 1)

if show_graphs:
    plt.figure(figsize=(20, 5))
    plt.subplot(2, 2, 1), plt.title("binary mask (2*std)"), plt.imshow(binary_mask, cmap='gray')
    plt.subplot(2, 2, 2), plt.title("erosion"), plt.imshow(erosion, cmap='gray')
    plt.subplot(2, 2, 3), plt.title("dilation"), plt.imshow(dilation, cmap='gray')
    plt.subplot(2, 2, 4), plt.title("erosion followed by dilation"), plt.imshow(noise_removed, cmap='gray')
    plt.show()

# Finding Contours 
# Create a copy of frame1 to draw contours and bounding box
frame1_copy = frame1.copy()
frame1_copy2 = frame1.copy()

# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(noise_removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(frame1_copy, contours, -1, (0, 255, 0), 3) 

x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(frame1_copy2,(x,y),(x+w,y+h),(0,255,0),2)

if show_graphs:
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1), plt.title("Binary Image"), plt.imshow(noise_removed, cmap='gray')
    plt.subplot(1, 3, 2), plt.title("Contours"), plt.imshow(frame1_copy, cmap='gray')
    plt.subplot(1, 3, 3), plt.title("Bounding Box"), plt.imshow(frame1_copy2, cmap='gray')
    plt.show()

# extract good features to track 
# Crop the region of interest (ROI) from the grayscale image
frame2_copy = frame2.copy()
roi = frame2_copy[y:y + h, x:x + w]

# Detect features in the ROI
corners = cv2.goodFeaturesToTrack(roi, 25, 0.01, 10)

if corners is not None:
     # Add bounding box offsets
    corners_global = corners + np.array([x, y], dtype=np.float32)
    for i in corners_global:
        # Adjust corner coordinates relative to the original image
        cx, cy = i.ravel()
        cx, cy = int(cx), int(cy)  # Add bounding box offsets
        #print(f"corner: ({cx}, {cy})")
        
        # Draw the feature point on the original frame
        cv2.circle(frame2_copy, (cx, cy), 3, (255, 0, 0), -1)
else:
    print("no corners have found")

if show_graphs:
    # Show the image with bounding box and features
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 1, 1), plt.title("Features in Bounding Box"), plt.imshow(frame2_copy, cmap='gray')
    plt.show()

# find these features in the next frame 
frame3_path = os.path.join(frames_folder, "0003.jpg")
frame3 = cv2.imread(frame3_path, cv2.IMREAD_GRAYSCALE)

# Create a mask image for drawing purposes
mask = np.zeros_like(frame2)
frame = frame3

# calculate optical flow
p0 = corners_global
p1, st, err = cv2.calcOpticalFlowPyrLK(frame2, frame3, p0, None)
# Select good points
if p1 is not None:
    good_new = p1[st==1]
    good_old = p0[st==1]
else:
    print("no corners have found with optical flow")

# draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 1)
    frame = cv2.circle(frame, (int(a), int(b)), 3, (255, 0, 0), 1)

if show_graphs:
    # Show the image with bounding box and features
    plt.figure(figsize=(20, 5))
    plt.subplot(1,2, 1), plt.title("Frame 2 Features in Bounding Box"), plt.imshow(frame2_copy, cmap='gray')
    plt.subplot(1, 2, 2), plt.title("Frame 3 Features in Bounding Box"), plt.imshow(frame, cmap='gray')
    plt.show()

# Initialize video writer
output_video_path = os.path.join(processed_data_folder, "tracked_video.avi")
frame_width = frame2.shape[1]
frame_height = frame2.shape[0]
fps = 10  # Adjust based on your video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Iterate through all frames in sorted order
frame_paths = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".jpg")])

# Persistent mask to store the tracks across all frames
mask = np.zeros_like(cv2.imread(frame_paths[0]))  # Use color mask (3 channels)

previous_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
p0 = corners_global  # Use features from the initial bounding box

for frame_path in frame_paths[1:]:
    # Read the current frame
    curr_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    curr_frame_color = cv2.imread(frame_path)  # For drawing in color

    # Calculate optical flow
    # not all points in p0 are successfully tracked in the next frame, 
    # the function returns p1 the position of the point in the next frame
    # and st a status array whene 1 indicate successful tracking and 0 tracking failed
    p1, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, curr_frame, p0, None)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        print(f"No corners found with optical flow for frame {frame_path}")
        continue

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        curr_frame_color = cv2.circle(curr_frame_color, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Combine the mask with the current frame
    output_frame = cv2.add(curr_frame_color, mask)

    # Write the frame to the output video
    out.write(output_frame)

    # Update for the next iteration
    previous_frame = curr_frame
    p0 = good_new.reshape(-1, 1, 2) # -1: Automatically infers the size of this dimension based on the total number of elements in this case number of points

# Release the video writer
out.release()
print(f"Output video saved at {output_video_path}")