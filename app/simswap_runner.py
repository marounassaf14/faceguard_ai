import subprocess
import shutil
import os
from pathlib import Path
from app.face_cropper import crop_faces
from app.frame_extractor import extract_frames
import random
import sys

def run_simswap(
    user_name,
    video_folder="simswap_new/videos",
    identities_dir="utils/faceswap_identities",
    output_folder="faces",
    simswap_dir="simswap_new",
    num_identities=3,
):
    # Step 1: Locate latest video
    video_search_path = Path(video_folder) / user_name
    user_videos = sorted(video_search_path.glob(f"{user_name}_*_clean.mp4"), key=os.path.getmtime)
    if not user_videos:
        print(f"❌ No clean video found for user: {user_name}")
        return

    latest_video = str(user_videos[-1])
    print(f"🎥 Using latest video: {latest_video}")

    # Step 2: Load and randomly select identity images
    identity_path = Path(identities_dir)
    identity_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        identity_images.extend(identity_path.glob(ext))

    if len(identity_images) < num_identities:
        print(f"❌ Not enough identity images found in {identities_dir} (found {len(identity_images)})")
        return

    identity_images = random.sample(identity_images, num_identities)
    print(f"🎯 Selected identities: {[img.name for img in identity_images]}")

    # Step 3: Set up paths
    simswap_script = Path(simswap_dir) / "test_video_swapsingle.py"
    arcface_model = Path(simswap_dir) / "arcface_model" / "arcface_checkpoint.tar"
    temp_path = Path(simswap_dir) / "temp_results"
    crop_size = "224"

    # Temporary directory to collect all fake frames
    tmp_fakes_dir = Path("tmp_fakes") / user_name
    tmp_fakes_dir.mkdir(parents=True, exist_ok=True)

    # Final output path for cropped fake faces
    fake_faces_dir = Path(output_folder) / user_name / "fake"
    fake_faces_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Run swaps
    for identity_image in identity_images:
        identity_name = identity_image.stem
        print(f"🔁 Swapping with identity: {identity_name}")
        python_exec = r"C:\Users\USER\anaconda3\envs\simswap\python.exe"

        identity_output_dir = Path("output") / user_name / identity_name
        os.makedirs(identity_output_dir, exist_ok=True)

        swapped_video_path = identity_output_dir / f"{identity_name}_swapped.mp4"
        identity_temp_path = temp_path / identity_name

        if identity_temp_path.exists():
            shutil.rmtree(identity_temp_path)
        os.makedirs(identity_temp_path, exist_ok=True)

        cmd = [
            python_exec,
            str(simswap_script.resolve()),
            "--crop_size", crop_size,
            "--use_mask",
            "--name", "people",
            "--Arc_path", str(arcface_model.resolve()),
            "--pic_a_path", str(identity_image.resolve()),
            "--video_path", str(Path(latest_video).resolve()),
            "--output_path", str(swapped_video_path.resolve()),
            "--gpu_ids", "-1",
            "--temp_path", str(identity_temp_path.resolve())
        ]

        result = subprocess.run(cmd, cwd=str(simswap_dir), capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ Error:\n", result.stderr)

        print(f"🎥 Saved swapped video for {identity_name} at: {swapped_video_path}")

        # Extract frames from swapped video and store in tmp_fakes_dir
        extracted_count = extract_frames(
            str(swapped_video_path),
            str(tmp_fakes_dir),
            step=5
        )
        print(f"🖼️ Extracted {extracted_count} frames to {tmp_fakes_dir}")

        # Clean temp
        shutil.rmtree(identity_temp_path)

    # Step 5: Crop all collected fake frames once at the end
    fake_index = len(list(fake_faces_dir.glob("*.jpg")))
    cropped_count = crop_faces(str(tmp_fakes_dir), str(fake_faces_dir), start_index=fake_index)
    print(f"👤 Cropped and saved {cropped_count} fake faces to {fake_faces_dir}")

    # Clean up temporary frame folder
    shutil.rmtree(tmp_fakes_dir)

    print(f"🎉 Finished all swaps for {user_name}")
