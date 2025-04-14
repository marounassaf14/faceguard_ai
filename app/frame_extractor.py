import cv2
import os

def extract_frames(video_path, output_folder, step=5):
    os.makedirs(output_folder, exist_ok=True)

    # Get starting index based on existing files
    existing_files = [f for f in os.listdir(output_folder) if f.endswith((".jpg", ".png"))]
    start_index = len(existing_files)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % step == 0:
            filename = os.path.join(output_folder, f"frame_{start_index + saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count
