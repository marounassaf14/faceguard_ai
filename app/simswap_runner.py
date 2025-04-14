# app/simswap_runner.py
import subprocess
import shutil
import os
from pathlib import Path
import glob
import sys

def run_simswap(user_name,
                video_folder="simswap_new/videos",
                identities_dir="utils/faceswap_identities",
                output_folder="faces",
                simswap_dir="simswap_new"):

    # Step 1: Get latest video for user
    video_search_path = Path(video_folder) / Path(user_name)
    print(video_search_path)
    user_videos = sorted(video_search_path.glob(f"{user_name}_*_clean.mp4"), key=os.path.getmtime)
    if not user_videos:
        print(f"❌ No clean video found for user: {user_name}")
        return

    latest_video = str(user_videos[-1])
    print(f"Using latest video: {latest_video}")

    # Step 2: Get all identity images
    identity_path = Path(identities_dir)
    identity_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        identity_images.extend(identity_path.glob(ext))
    if not identity_images:
        print(f"❌ No identity images found in {identities_dir}")
        return
    
    

    # Step 3: Prepare paths
    simswap_script = Path(simswap_dir) / "test_video_swapsingle.py"
    arcface_model = Path(simswap_dir) / "arcface_model" / "arcface_checkpoint.tar"
    temp_path = Path(simswap_dir) / "temp_results"
    crop_size = "224"

    final_output_path = Path(output_folder) / user_name / "fake"
    os.makedirs(final_output_path, exist_ok=True)
    existing_fakes = list(final_output_path.glob("*.jpg"))
    fake_index = len(existing_fakes)

    print(latest_video)

    # Step 4: Loop through identities and swap
    for identity_image in identity_images:
        identity_name = identity_image.stem
        print(f"🔁 Swapping with identity: {identity_name}")
        python_exec = r"C:\Users\USER\anaconda3\envs\simswap\python.exe"
        print(f"▶️ Using Python for subprocess: {python_exec}")


        cmd = [
            python_exec,
            str(simswap_script.resolve()),
            "--crop_size", crop_size,
            "--use_mask",
            "--name", "people",
            "--Arc_path", str(arcface_model.resolve()),
            "--pic_a_path", str(identity_image.resolve()),
            "--video_path", str(Path(latest_video).resolve()),
            "--output_path", f"{identity_name}_swapped.mp4",
            "--gpu_ids", "-1",
            "--temp_path", str(temp_path.resolve())
        ]

        result = subprocess.run(
            cmd,
            cwd=str(simswap_dir),  # 👈 sets working directory
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print("⚠️ Error:\n", result.stderr)

        # Move new generated frames
        new_frames = list(temp_path.glob("*.jpg"))
        for frame in new_frames:
            target_file = final_output_path / f"deepfake_{fake_index:04d}.jpg"
            shutil.move(str(frame), str(target_file))
            fake_index += 1
        print(f"✅ Appended {len(new_frames)} fakes to {final_output_path}")
