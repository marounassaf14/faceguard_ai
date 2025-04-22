import gradio as gr
from app.ui import process_video, delete_user_data, detect_deepfake

with gr.Blocks(title="FaceGuard AI") as demo:
    gr.Markdown("# 🛡️ FaceGuard AI")
    gr.Markdown("Choose your mode: create dataset or detect deepfake in a video.")

    # Mode selection
    mode = gr.Radio(["Create Dataset", "Detect Deepfake"], label="Select Mode", value="Create Dataset")

    with gr.Row():
        video_input = gr.Video(label="Upload or Record a Video")
        user_input = gr.Textbox(label="Username", placeholder="e.g. maroun", visible=True)

    run_btn = gr.Button("▶️ Run")

    # Output areas
    output_text = gr.Textbox(label="Logs", lines=15, visible=True)
    video_output = gr.Video(label="Processed Video Output", visible=False)

    # Delete panel (initially visible)
    delete_section = gr.Accordion("🗑️ Delete User Data", open=False, visible=True)
    with delete_section:
        delete_input = gr.Textbox(label="Enter username to delete", placeholder="e.g. maroun")
        delete_btn = gr.Button("Delete All User Data")
        delete_output = gr.Textbox(label="Deletion Result")
        delete_btn.click(delete_user_data, inputs=delete_input, outputs=delete_output)

    # Function to dynamically show/hide UI parts
    def toggle_outputs(selected_mode):
        return (
            gr.update(visible=(selected_mode == "Create Dataset")),  # username
            gr.update(visible=(selected_mode == "Create Dataset")),  # logs
            gr.update(visible=(selected_mode == "Detect Deepfake")),  # video output
            gr.update(visible=(selected_mode == "Create Dataset"))   # delete panel
        )

    mode.change(toggle_outputs, inputs=mode, outputs=[user_input, output_text, video_output, delete_section])

    # Run logic
    def handle_action(selected_mode, video, username):
        if selected_mode == "Create Dataset":
            logs = process_video(video, username)
            return logs, None
        else:
            output_path = detect_deepfake(video)
            return None, output_path

    run_btn.click(
        handle_action,
        inputs=[mode, video_input, user_input],
        outputs=[output_text, video_output]
    )

if __name__ == "__main__":
    demo.launch()
