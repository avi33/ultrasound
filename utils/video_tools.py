import cv2
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default='', type=str)
    parser.add_argument("--gif-path", default='', type=str)  
    parser.add_argument("--resize-factor", default=1, type=float)
    parser.add_argument("--frame-interval", default=1, type=int)
    args = parser.parse_args()
    return args

def mp4_to_gif(video_path, gif_path, resize_factor=1, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_count += 1

    cap.release()

    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=100)

if __name__ == "__main__":
    args = parse_args()
    mp4_to_gif(args.video_path, args.gif_path, args.resize_factor, args.frame_interval)