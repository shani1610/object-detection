import os
import cv2
from moviepy import VideoFileClip

class VideoTools:
    def __init__(self, source=None):
        self.source = source
        self.extension = None  # Default to None if not provided

    @staticmethod
    def extract_frames(source, output_path, num_frames=None):
        """
        Extract frames from a video and save them to the output directory.
        Args:
            source (str): Path to the video file.
            output_path (str): Directory to save the frames.
            num_frames (int, optional): Number of frames to extract. Defaults to all frames.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Open video using OpenCV
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        if num_frames is None or num_frames > total_frames:
            num_frames = total_frames

        frame_interval = total_frames // num_frames if num_frames else 1

        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame only if it matches the extraction interval
            if frame_count % frame_interval == 0:
                frame_name = os.path.join(output_path, f"{saved_count:04d}.jpg")
                cv2.imwrite(frame_name, frame)
                saved_count += 1
                if saved_count >= num_frames:
                    break

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames to {output_path}")

    @staticmethod
    def gif2mp4(source, output_path):
        """
        Convert a GIF to MP4 format.
        Args:
            source (str): Path to the GIF file.
            output_path (str): Path to save the MP4 file.
        """
        if not source.lower().endswith(".gif"):
            raise ValueError("Source file is not a GIF")

        clip = VideoFileClip(source)
        clip.write_videofile(output_path, codec="libx264")
        print(f"Converted {source} to MP4 at {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VideoTools: Extract frames and convert GIFs to MP4")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Subparser for extract_frames
    extract_parser = subparsers.add_parser("extract_frames", help="Extract frames from a video")
    extract_parser.add_argument("source", type=str, help="Path to the video file")
    extract_parser.add_argument("output_path", type=str, help="Path to save the frames")
    extract_parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to extract (optional)")

    # Subparser for gif2mp4
    gif_parser = subparsers.add_parser("gif2mp4", help="Convert a GIF to MP4")
    gif_parser.add_argument("source", type=str, help="Path to the GIF file")
    gif_parser.add_argument("output_path", type=str, help="Path to save the MP4 file")

    args = parser.parse_args()

    if args.command == "extract_frames":
        VideoTools.extract_frames(args.source, args.output_path, args.num_frames)
    elif args.command == "gif2mp4":
        VideoTools.gif2mp4(args.source, args.output_path)
    else:
        parser.print_help()