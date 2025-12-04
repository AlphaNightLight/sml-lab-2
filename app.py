import gradio as gr
import json, time, os
from huggingface_hub import HfApi
from llama_cpp import Llama

MODEL_REPO = "fedealex/llama-1B"       # The model is on hf
MODEL_FILE = "model-1b-q8_0.gguf"      # Name of the model file
DATASET_REPO = "fedealex/flags"        # Flages saved on hf
HF_TOKEN = os.getenv("HF_TOKEN")       # To access hf
LOCAL_FLAGS = "flags.json"             # You must save locally before push to hf



print("Loading model...")

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_threads=2,
    n_batch=128,
    n_ctx=2048,
    temperature=0.7
)

### The Chat Model
def chat_model(message, history):

    # Retrieve the context
    prompt = ""
    for item in history:
        role = item["role"]
        text = item["content"][0]["text"]
        prompt += f"<|{role}|>{text}\n"
    prompt += f"<|user|>{message}\n<|assistant|>"

    # Invoke the model
    output = llm(prompt, max_tokens=350)
    return output["choices"][0]["text"].strip()



### Save the flags
def save_flag_to_dataset(history, reason):
    # The record to be submitted
    record = {
        "timestamp": time.time(),
        "history": history,
        "reason": reason
    }

    # First we save it locally
    with open(LOCAL_FLAGS, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Then we send to the hf dataset
    api = HfApi()
    api.upload_file(
        path_or_fileobj=LOCAL_FLAGS,
        path_in_repo=LOCAL_FLAGS,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN
    )

    if reason=="GOOD":
        return "Feedback reported successfully!"
    else:
        return "Flag reported successfully!"



### The Gradio App
with gr.Blocks() as app:
    # Title
    gr.Markdown("### Scalable Machine Learning Lab 2")

    # Chat Box
    chat_box = gr.ChatInterface(
        fn=chat_model,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="How can I help you today?"),
        title="Llama Finetuned",
        description="You are using the model: "+MODEL_REPO+"/"+MODEL_FILE
    )

    # Feedback Buttons
    gr.Markdown("### Let us know what do you think of our chatbot!")
    good_btn = gr.Button("Appreciate conversation ❤", variant="huggingface")
    flag_btn = gr.Button("Flag Conversation", variant="stop")

    # We allow the user to select flagging reason
    with gr.Group(visible=False) as flag_group:
        gr.Markdown("### What kind of problem are you facing?")
        reason_dd = gr.Dropdown(
            choices=[
                "Offensive / Toxic",
                "Incorrect Output",
                "Hallucination",
                "Safety Concern",
                "Biased Output",
                "Other"
            ],
            label="Flagging Reason"
        )
        submit_flag_btn = gr.Button("Submit Flag", variant="primary")
        cancel_flag_btn = gr.Button("Cancel")

    # To inform the user about feedback status
    feedback_status = gr.Textbox(label="Feedback Status", visible=True)

    # Button callbacks
    flag_btn.click(
        lambda: gr.update(visible=True),
        inputs=None,
        outputs=flag_group
    )

    cancel_flag_btn.click(
        lambda: gr.update(visible=False),
        inputs=None,
        outputs=flag_group
    )

    submit_flag_btn.click(
        lambda history, reason: save_flag_to_dataset(history, reason),
        inputs=[chat_box.chatbot, reason_dd],
        outputs=feedback_status
    ).then(
        lambda: gr.update(visible=False), None, flag_group
    )
    
    dummy_markdown = gr.Markdown("GOOD", visible=False) # To be able to pass a string as a gradio block
    good_btn.click(
        lambda history, reason: save_flag_to_dataset(history, reason),
        inputs=[chat_box.chatbot, dummy_markdown],
        outputs=feedback_status
    )



app.launch()
