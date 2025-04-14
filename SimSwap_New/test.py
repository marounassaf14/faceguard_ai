from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import glob

image_filenames = sorted(glob.glob("./tmp/*.jpg"))
clip = ImageSequenceClip(image_filenames, fps=15.0)
clip = clip.set_duration(len(image_filenames) / 15.0)
clip = clip.without_audio()
clip.write_videofile("./output/test.mp4", codec="libx264", fps=15.0, audio=False)
