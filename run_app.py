import gradio as gr
from app.ui import process_video
from app.ui import delete_user_data

with gr.Blocks(title="FaceGuard AI – Deepfake Generator") as demo:
    gr.Markdown("# 🛡️ FaceGuard AI – Deepfake Generator")
    gr.Markdown("Extracts frames, crops faces, and uses SimSwap to generate deepfake images.")

    with gr.Row():
        video_input = gr.Video(label="Upload or Record a Video")
        user_input = gr.Textbox(label="Username", placeholder="e.g. maroun")

    run_btn = gr.Button("▶️ Process Video")
    output_text = gr.Textbox(label="Logs", lines=15)

    run_btn.click(process_video, inputs=[video_input, user_input], outputs=output_text)

    with gr.Accordion("🗑️ Delete User Data", open=False):
        delete_input = gr.Textbox(label="Enter username to delete", placeholder="e.g. maroun")
        delete_btn = gr.Button("Delete All User Data")
        delete_output = gr.Textbox(label="Deletion Result")

        delete_btn.click(delete_user_data, inputs=delete_input, outputs=delete_output)

if __name__ == "__main__":
    demo.launch()
