# src/app.py

import gradio as gr
from predict_ticket import predict_ticket

# Load product list (you can also load from file)
PRODUCT_LIST = ["EcoBreeze AC", "SmartWatch V2", "PhotoSnap Cam", "SoundWave 300"]

# Main prediction wrapper
def classify_ticket(ticket_text):
    result = predict_ticket(ticket_text, PRODUCT_LIST)
    return (
        result["issue_type"],
        result["urgency_level"],
        result["extracted_entities"]
    )

# Gradio UI
def launch_app():
    with gr.Blocks(theme=gr.themes.Soft(), css="footer {visibility: hidden}") as demo:
        gr.Markdown("""
        # 🛠️ AI Ticket Classifier
        Welcome to the intelligent customer support ticket analyzer.
        Paste your ticket below and get instant predictions!
        """)

        with gr.Row():
            ticket_input = gr.Textbox(
                label="🎫 Enter Ticket Text",
                placeholder="Describe your issue...",
                lines=5,
                show_label=True
            )

        with gr.Row():
            submit_btn = gr.Button("🚀 Predict")

        with gr.Row():
            issue_output = gr.Textbox(label="📌 Predicted Issue Type")
            urgency_output = gr.Textbox(label="⚡ Predicted Urgency Level")

        with gr.Accordion("📎 Extracted Entities", open=False):
            entity_output = gr.JSON(label="Entities (Product, Dates, Complaints)")

        submit_btn.click(
            fn=classify_ticket,
            inputs=[ticket_input],
            outputs=[issue_output, urgency_output, entity_output]
        )

        gr.Markdown("""
        ---
        Made with ❤️ using Gradio + Sklearn + NLP
        """)

    demo.launch()

if __name__ == "__main__":
    launch_app()
