import gradio as gr
import torch
from transformers import pipeline
from prometheus_client import start_http_server, Counter, Summary
import time

pipe = pipeline("text-generation", "distilgpt2", device='cpu')

REQUEST_COUNT = Counter('chatbot_requests_total', 'Total number of chatbot requests')
REQUEST_LATENCY = Summary('chatbot_request_latency_seconds', 'Latency of chatbot requests')

stop_inference = False

def respond(
    message,
    history: list[tuple[str, str]],
    system_message="You are a friendly Chatbot.",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
):
    global stop_inference
    stop_inference = False
    REQUEST_COUNT.inc()
    start_time = time.time()

    if history is None:
        history = []

    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    response = ""
    outputs = pipe(
        message,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
    )

    response = outputs[0]['generated_text']
    REQUEST_LATENCY.observe(time.time() - start_time)
    yield history + [(message, response)]

def cancel_inference():
    global stop_inference
    stop_inference = True

custom_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}

.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}

.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #45a049;
}

.gr-slider input {
    color: #4CAF50;
}

.gr-chat {
    font-size: 16px;
}

#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
    gr.Markdown("Interact with the AI chatbot using customizable settings below.")

    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")

    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")

    cancel_button = gr.Button("Cancel Inference", variant="danger")

    user_input.submit(respond, [user_input, chat_history, system_message, max_tokens, temperature, top_p], chat_history)

    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    start_http_server(8000)
    demo.launch(share=False)
