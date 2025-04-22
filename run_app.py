import gradio as gr
from app.ui import process_video, delete_user_data
from app.load_models import (
    facenet_embedder,
    real_embs_tensor,
    real_labels,
    fake_embs_tensor,
    fake_origins,
    fake_swaps,
    xception_model,
    device
)
from app.deepfake_detector import detect_deepfake, detect_image_deepfake

with gr.Blocks(title="FaceGuard AI", css="""
    #styled_output textarea {
        font-size: 20px !important;
        text-align: center !important;
        font-weight: bold;
        line-height: 1.6;
    }
""") as demo:
    gr.Markdown("# 🛡️ FaceGuard AI")
    gr.Markdown("Choose your mode: create dataset or detect deepfake in a video or image.")

    # Mode selection
    mode = gr.Radio(
        ["Create Dataset", "Detect Deepfake", "Detect Deepfake (Image)"],
        label="Select Mode",
        value="Create Dataset"
    )

    # Inputs
    video_input = gr.Video(label="Upload or Record a Video", visible=True)
    image_input = gr.Image(type="filepath", label="Upload a Cropped Face Image", visible=False)
    user_input = gr.Textbox(label="Username", placeholder="e.g. maroun", visible=True)

    run_btn = gr.Button("▶️ Run")

    # Output areas
    output_text = gr.Textbox(label="Logs / Results", lines=15, visible=True)
    video_output = gr.Video(label="Processed Video Output", visible=False)
    image_output_text = gr.Textbox(
        label="Image Analysis Result",
        lines=6,
        visible=False,
        elem_id="styled_output"
    )

    # Delete section
    delete_section = gr.Accordion("🗑️ Delete User Data", open=False, visible=True)
    with delete_section:
        delete_input = gr.Textbox(label="Enter username to delete", placeholder="e.g. maroun")
        delete_btn = gr.Button("Delete All User Data")
        delete_output = gr.Textbox(label="Deletion Result")
        delete_btn.click(delete_user_data, inputs=delete_input, outputs=delete_output)

    # Dynamic component visibility
    def toggle_outputs(selected_mode):
        return (
            gr.update(visible=(selected_mode != "Detect Deepfake (Image)")),  # video_input
            gr.update(visible=(selected_mode == "Detect Deepfake (Image)")),  # image_input
            gr.update(visible=(selected_mode == "Create Dataset")),           # username
            gr.update(visible=(selected_mode == "Create Dataset")),           # logs
            gr.update(visible=(selected_mode == "Detect Deepfake")),          # video_output
            gr.update(visible=(selected_mode == "Create Dataset")),           # delete_section
            gr.update(visible=(selected_mode == "Detect Deepfake (Image)"))   # image_output_text
        )

    mode.change(
        toggle_outputs,
        inputs=mode,
        outputs=[video_input, image_input, user_input, output_text, video_output, delete_section, image_output_text]
    )

    # Main handler
    def handle_action(selected_mode, video, username, image):
        if isinstance(video, dict) and "name" in video:
            video = video["name"]

        if selected_mode == "Create Dataset":
            logs = process_video(video, username)
            return logs, gr.update(visible=False), gr.update(visible=False)

        elif selected_mode == "Detect Deepfake":
            output_path = detect_deepfake(
                video,
                facenet_embedder,
                real_embs_tensor, real_labels,
                fake_embs_tensor, fake_origins, fake_swaps,
                xception_model, device
            )
            return None, gr.update(value=output_path, visible=True), gr.update(visible=False)

        elif selected_mode == "Detect Deepfake (Image)":
            _, result_text = detect_image_deepfake(
                image,
                facenet_embedder,
                real_embs_tensor, real_labels,
                fake_embs_tensor, fake_origins, fake_swaps,
                xception_model, device
            )
            return result_text, gr.update(visible=False), gr.update(value=result_text, visible=True)

    # Button click event
    run_btn.click(
        handle_action,
        inputs=[mode, video_input, user_input, image_input],
        outputs=[output_text, video_output, image_output_text]
    )

if __name__ == "__main__":
    demo.launch()
