# run_app.py
import gradio as gr
from app.ui import process_video

iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Upload or Record a Video"),
        gr.Textbox(label="Username", placeholder="e.g. maroun")
    ],
    outputs="text",
    title="FaceGuard AI – Deepfake Generator",
    description="This tool extracts frames, crops faces, and uses SimSwap to generate deepfake images."
)

if __name__ == "__main__":
    iface.launch()
