import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import json

def get_model_params(name):
    with open("model_params.json", "r") as f:
        data = json.load(f)
    return data[name]

def make_params_components(name):
    params = get_model_params(name)
    values = {}
    for key, param in params.items():
        if param["type"] == "slider":
            values[key] = st.slider(
                label=param["label"], 
                min_value=param["min"], 
                max_value=param["max"], 
                value=param["default"], 
                step=param["step"],
                help=param["description"]
            )
        elif param["type"] == "text":
            values[key] = [st.text_input(
                label=param["label"],
                value=param["default"],
                help=param["description"]
            )]
    return values

st.title("Llama/LangChain Local Chatbot")

st.sidebar.header("Settings")
model_options = ["llama3.2", "deepseek-r1:1.5b"]
MODEL = st.sidebar.selectbox("Choose a Model", model_options, index=0, on_change=lambda: st.session_state.__setitem__("messages",[]))
with st.sidebar.expander("Advanced Settings", expanded=False):
    params = make_params_components(MODEL)

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOllama(model=MODEL)

for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

def join_messages(messages):
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    return chat_history

def stream_answering(prompt_with_history, container):
    reply = ""
    stream = llm.stream(prompt_with_history)
    for chunk in stream:
        reply += chunk.text()
        container.markdown(reply)
    return reply

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)

    if MODEL == "deepseek-r1:1.5b":
        # --> step for no streaming (real-time answering)
        st.session_state.messages.append(HumanMessage(content=prompt, type='human'))
        response = llm.invoke(st.session_state.messages)
        reply = response.content
        reply = str(reply).replace("<think>", "").replace("</think>", "")
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append(AIMessage(content=reply, type='ai'))

    elif MODEL == "llama3.2":
        # --> step for streaming (real-time answering)
        message_placeholder = st.chat_message("assistant").empty()
        chat_history = join_messages(st.session_state.messages)
        if chat_history:
            final_prompt = f"Our current conversation:\n{chat_history}\n\nI ask this: {prompt}"
        else:
            final_prompt = f"I ask this: {prompt}"

        reply = stream_answering(final_prompt, message_placeholder)
        st.session_state.messages.append(HumanMessage(content=prompt, type='human'))
        st.session_state.messages.append(AIMessage(content=reply, type='ai'))

        # --> debugging prompts
        # print()
        # print(final_prompt)

with st.sidebar.expander(label="Chat Summary"):
    if st.button(label="Summarize Chat History"):
        chat_history = join_messages(st.session_state.messages)
        summary_container = st.empty()
        if chat_history:
            final_prompt = f"Our current conversation:\n{chat_history}\n\nI ask this: Can you help me summarize our conversation?"
            reply = stream_answering(final_prompt, summary_container)
        else:
            summary_container.warning("You haven't made any conversation...")