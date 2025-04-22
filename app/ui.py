import os
import shutil
import tempfile
from datetime import datetime
from app.frame_extractor import extract_frames
from app.face_cropper import crop_faces
from app.simswap_runner import run_simswap
from subprocess import run

def reencode_video(input_path, output_path):
    print(f"🔁 Re-encoding with ffmpeg: {input_path} ➡️ {output_path}")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        output_path
    ]
    result = run(cmd, capture_output=True)
    if result.returncode != 0:
        print("❌ FFmpeg error:\n", result.stderr.decode())
    else:
        print("✅ Re-encoding complete.")

def process_video(video, user_name):
    logs = []

    # Temp copy for processing
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, f"{user_name}.mp4")
    shutil.copy(video, temp_video_path)

    # Save permanent raw video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_video_dir = f"simswap_new/videos/{user_name}"
    os.makedirs(user_video_dir, exist_ok=True)
    saved_video_path = os.path.join(user_video_dir, f"{user_name}_{timestamp}.mp4")
    shutil.copy(temp_video_path, saved_video_path)
    logs.append(f"✅ Video saved as: {saved_video_path}")

    # Re-encode for moviepy compatibility
    reencoded_path = saved_video_path.replace(".mp4", "_clean.mp4")
    reencode_video(saved_video_path, reencoded_path)
    logs.append(f"🎞️ Re-encoded video saved as: {reencoded_path}")

    # Frame extraction
    frame_dir = f"frames/{user_name}"
    os.makedirs(frame_dir, exist_ok=True)
    existing_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    new_frame_count = extract_frames(reencoded_path, frame_dir, step=5)
    logs.append(f"🖼️ {new_frame_count} new frames extracted to {frame_dir}")

    # Face cropping
    face_dir = f"faces/{user_name}/real"
    os.makedirs(face_dir, exist_ok=True)
    existing_faces = len([f for f in os.listdir(face_dir) if f.endswith(".jpg")])
    new_face_count = crop_faces(frame_dir, face_dir, start_index=existing_frames)
    logs.append(f"👤 {new_face_count} new faces cropped to {face_dir}")

    # Deepfake generation
    deepfake_dir = f"faces/{user_name}/fake"
    os.makedirs(deepfake_dir, exist_ok=True)
    existing_fakes = len([f for f in os.listdir(deepfake_dir) if f.endswith(".jpg")])
    run_simswap(user_name)
    final_fakes = len([f for f in os.listdir(deepfake_dir) if f.endswith(".jpg")])
    new_fake_count = final_fakes - existing_fakes
    logs.append(f"🧪 {new_fake_count} deepfake images generated in {deepfake_dir}")

    return "\n".join(logs)

def detect_deepfake():
    pass

# 🗑️ Utility to delete user data
def delete_user_data(user_name):
    if not user_name.strip():
        return "⚠️ Please enter a valid username."

    paths = [
        f"frames/{user_name}",
        f"faces/{user_name}",
        f"simswap_new/videos/{user_name}"
    ]
    deleted = []
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            deleted.append(path)

    return f"✅ Deleted: {deleted}" if deleted else "⚠️ No data found for that user."
