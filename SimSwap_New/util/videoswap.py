'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os 
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import  time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo=False, use_mask=False):
    from PIL import Image
    from moviepy.editor import AudioFileClip
    import subprocess

    # Check if video has audio
    video_forcheck = VideoFileClip(video_path)
    has_audio = video_forcheck.audio is not None
    if has_audio:
        audio_path = "./temp_audio.aac"
        video_forcheck.audio.write_audiofile(audio_path, codec='aac')
    del video_forcheck

    # Prepare OpenCV capture
    video = cv2.VideoCapture(video_path)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0.0 or fps is None:
        print("FPS undetected, defaulting to 25")
        fps = 25
    print(f"Detected FPS: {fps}")

    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir)

    spNorm = SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes).cuda()
        net.load_state_dict(torch.load('./parsing_model/checkpoint/79999_iter.pth'))
        net.eval()
    else:
        net = None

    for frame_index in tqdm(range(frame_count)):
        ret, frame = video.read()
        if not ret:
            break

        detect_results = detect_model.get(frame, crop_size)
        if detect_results is not None:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tenor_list = []

            for frame_align_crop in frame_align_crop_list:
                frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[None, ...].cuda()
                swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                swap_result_list.append(swap_result)
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)

            reverse2wholeimage(
                frame_align_crop_tenor_list,
                swap_result_list,
                frame_mat_list,
                crop_size,
                frame,
                logoclass,
                os.path.join(temp_results_dir, f'frame_{frame_index:07d}.jpg'),
                no_simswaplogo,
                pasring_model=net,
                use_mask=use_mask,
                norm=spNorm
            )
        else:
            if not no_simswaplogo:
                frame = logoclass.apply_frames(frame)
            cv2.imwrite(os.path.join(temp_results_dir, f'frame_{frame_index:07d}.jpg'), frame)

    video.release()

    # === Combine frames with OpenCV ===
    print("Merging frames into final video...")
    frame_paths = sorted(glob.glob(os.path.join(temp_results_dir, '*.jpg')))
    if not frame_paths:
        print("❌ No frames found!")
        return

    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = save_path.replace('.mp4', '_noaudio.mp4')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    for path in frame_paths:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    print(f"Video frames saved to {temp_video_path}")

    # === Add audio if it exists ===
    if has_audio:
        print("Adding original audio...")
        final_cmd = f'ffmpeg -y -i "{temp_video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{save_path}"'
        subprocess.call(final_cmd, shell=True)
        os.remove(temp_video_path)
        os.remove(audio_path)
        print(f"Final video with audio saved to {save_path}")
    else:
        os.rename(temp_video_path, save_path)
        print(f"Final video saved to {save_path} (no audio)")
