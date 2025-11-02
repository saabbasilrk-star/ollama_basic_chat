import sys
import hyperdiv as hd
import requests
from ollama import Client

ollama_url = 'http://localhost:11434'
model_list = []
client = None

def initialize_ollama():
    global client, model_list
    try:
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            client = Client(host=ollama_url)
            api_return = client.list()
            
            print("Available models:", api_return)  # Debug print
            
            if isinstance(api_return, dict) and 'models' in api_return:
                model_list = [model.get('name') for model in api_return['models'] if model.get('name')]
                if not model_list:
                    print("No models found. Please download a model using 'ollama pull modelname'")
                    sys.exit(1)
            else:
                print("Invalid API response format")
                sys.exit(1)
        else:
            print(f"Ollama server error: {response.status_code}")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {str(e)}")
        print("Please ensure Ollama is running with 'ollama serve'")
        sys.exit(1)

# Call initialization
initialize_ollama()

def add_message(role, content, state, gpt_model):
    state.messages += (
        dict(role=role, content=content, id=state.message_id, gpt_model=gpt_model),
    )
    state.message_id += 1

def request(gpt_model, state):
    if not client:
        state.current_reply = "Error: Ollama client not initialized"
        return
        
    try:
        if not gpt_model:
            raise ValueError("No model selected")
            
        messages = [dict(role=m["role"], content=m["content"]) for m in state.messages]
        response = client.chat(
            model=gpt_model,
            messages=messages,
            stream=True,
        )

        state.current_reply = ""  # Reset reply at start
        for chunk in response:
            if isinstance(chunk, dict) and 'message' in chunk:
                content = chunk['message'].get('content', '')
                if content:
                    state.current_reply += content
            else:
                print(f"Unexpected chunk format: {chunk}")  # Debug print

        if state.current_reply:
            add_message("assistant", state.current_reply, state, gpt_model)
        state.current_reply = ""
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        state.current_reply = error_msg
        print(f"Chat request failed: {error_msg}")

def render_user_message(content, gpt_model):
    with hd.hbox(
        align="center",
        padding=0.5,
        border_radius="medium",
        background_color="neutral-50",
        font_color="neutral-600",
        justify="space-between",
    ):
        with hd.hbox(gap=0.5, align="center"):
            hd.icon("chevron-right", shrink=0)
            hd.text(content)
        hd.badge(gpt_model)

def main():
    state = hd.state(messages=(), current_reply="", gpt_model="gpt-4", message_id=0)
    task = hd.task()
    template = hd.template(title="Ollama Basic Chatbot", sidebar=False)

    with template.body:
        if len(state.messages) > 0:
            with hd.box(direction="vertical-reverse", gap=1.5, vertical_scroll=True):
                if state.current_reply:
                    hd.markdown(state.current_reply)

                for e in reversed(state.messages):
                    with hd.scope(e["id"]):
                        if e["role"] == "system":
                            continue
                        if e["role"] == "user":
                            render_user_message(e["content"], e["gpt_model"])
                        else:
                            hd.markdown(e["content"])

        with hd.box(align="center", gap=1.5):
            with hd.form(direction="horizontal", width="100%") as form:
                with hd.box(grow=1):
                    prompt = form.text_input(
                        placeholder="Talk to Ollama",
                        autofocus=True,
                        disabled=task.running,
                        name="prompt",
                    )

                model = form.select(
                    options=model_list,
                    value=model_list[0] if model_list else None,  # Use first available model
                    name="gpt-model",
                    placeholder='Select a model'
                )

            if form.submitted:
                if prompt.value:  # Ensure prompt is not empty
                    add_message("user", prompt.value, state, model.value)
                    prompt.reset()
                    task.rerun(request, model.value, state)

            if len(state.messages) > 0:
                if hd.button(
                    "Start Over", size="small", variant="text", disabled=task.running
                ).clicked:
                    state.messages = ()

hd.run(main)
