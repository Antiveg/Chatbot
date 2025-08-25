import streamlit as st
from openai import OpenAI, RateLimitError, APIError
from openai.types.chat import ChatCompletionMessageParam
from typing import cast, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfigDict
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InvalidArgument, GoogleAPICallError, RetryError
import json

# get client from openai with api_key
try:
    openai_api_key = st.secrets["openai"]["api_key"]
    openai_client = OpenAI(api_key=openai_api_key)
except:
    st.error("⚠️ OpenAI API key missing or your secrets.toml is invalid!")
    st.stop()

try:
    gemini_api_key = st.secrets["gemini"]["api_key"]
    genai.configure(api_key=gemini_api_key)
except:
    st.error("⚠️ Gemini API key missing or your secrets.toml is invalid!")
    st.stop()

# functions for sidebar setup
def get_model_names():
    names = {}

    openai_models = openai_client.models.list()
    for m in openai_models:
        if m.id.startswith(("gpt-3.5", "gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4", "davinci", "babbage")):
            names[m.id] = "openai"

    gemini_models = genai.list_models()
    for m in gemini_models:
        if any(keyword in str(m.name) for keyword in ("pro", "flash", "live", "dialog", "chat", "thinking", "tts", "exp", "lite")):
            names[m.name] = "gemini"

    return names

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
            values[key] = st.text_input(
                label=param["label"],
                value=param["default"],
                help=param["description"]
            )
    return values

# sidebar model configuration
st.sidebar.title("Configuration")
models = get_model_names()
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()), on_change=lambda: st.session_state.__setitem__("messages", []))
with st.sidebar.expander("Advanced Settings", expanded=False):
    params = make_params_components(models[model_name])

gemini_model = None
if models[model_name] == "gemini":
    gemini_model = genai.GenerativeModel(
        model_name=str(model_name),
        generation_config=cast(GenerationConfigDict, params)
    )

# chat history state management
st.title("OpenAI/Gemini Chatbot")
if "messages" not in st.session_state:
    if models[model_name] == "openai":
        st.session_state.messages = [
            {"role":"system","content":"Please respond concisely, straight to the point"}
        ]
    elif models[model_name] == "gemini":
        st.session_state.messages = [
            {"role":"system","parts":["Please respond concisely, straight to the point"]}
        ]

for message in st.session_state.messages:
    if message['role'] == "system": continue
    with st.chat_message(message['role']):
        if models[model_name] == "openai":
            st.markdown(message["content"])
        elif models[model_name] == "gemini":
            st.markdown(message["parts"][0])

# chat flow (something similar to event-driven)
def openai_chat(model_name : str):
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=cast(List[ChatCompletionMessageParam], st.session_state.messages),
            **params
        )
        reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    except RateLimitError:
        st.error("⚠️ You've hit your usage limit or quota. Please check your billing.")
    except APIError:
        st.error("⚠️ The service is currently unavailable. Try again later.")
    except Exception as e:
        st.error("⚠️ An unexpected error occurred. Please try again.")

def gemini_chat(model_name : str):
    try:
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if gemini_model:
            response = gemini_model.generate_content(contents=st.session_state.messages)
            reply = response.text

            st.session_state.messages.append({"role": "model", "parts": [reply]})
            with st.chat_message("model"):
                st.markdown(reply)

    except ResourceExhausted:
        st.error("⚠️ You've hit your usage limit or quota. Please check your Gemini billing.")
    except ServiceUnavailable:
        st.error("⚠️ Gemini service is currently unavailable. Try again later.")
    except InvalidArgument as e:
        st.error(f"⚠️ Invalid request: {str(e)}")
    except (GoogleAPICallError, RetryError) as e:
        st.error(f"⚠️ Gemini API error: {str(e)}")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {str(e)}")
    
if prompt := st.chat_input("Say something..."):
    if models[model_name] == "openai":
        openai_chat(model_name=str(model_name))
    elif models[model_name] == "gemini":
        gemini_chat(model_name=str(model_name))

# Button to trigger popup
summary_placeholder = st.sidebar.empty()
if st.sidebar.button("Summarize Chat History"):
    try:
        chat_history = ""
        if models[model_name] == "openai":
            chat_history = ". ".join(
                [f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages]
            )
            response = openai_client.chat.completions.create(
                model=str(model_name),
                messages=[
                    {"role": "system", "content": "Summarize the following chat history"},
                    {"role": "user", "content": chat_history}
                ],
                **params
            )
            summary = response.choice[0].message.content
            summary_placeholder.info(summary)
        elif models[model_name] == "gemini":
            chat_history = ". ".join(
                [f"{msg["role"]}: {msg["parts"][0]}" for msg in st.session_state.messages]
            )
            if gemini_model:
                response = gemini_model.generate_content(
                    contents=[
                        {'role':'model','parts':['Summarize the following chat history at max 100 words']},
                        {'role':'user','parts':[chat_history]},
                    ]
                )
                summary = response.text
                summary_placeholder.info(summary)
    except RateLimitError:
        summary_placeholder.error("⚠️ You've hit your usage limit or quota. Please check your billing.")
    except APIError:
        summary_placeholder.error("⚠️ The service is currently unavailable. Try again later.")
    except ResourceExhausted:
        summary_placeholder.error("⚠️ You've hit your usage limit or quota. Please check your Gemini billing.")
    except ServiceUnavailable:
        summary_placeholder.error("⚠️ Gemini service is currently unavailable. Try again later.")
    except InvalidArgument as e:
        summary_placeholder.error(f"⚠️ Invalid request: {str(e)}")
    except (GoogleAPICallError, RetryError) as e:
        summary_placeholder.error(f"⚠️ Gemini API error: {str(e)}")
    except Exception as e:
        summary_placeholder.error(f"⚠️ An unexpected error occurred: {str(e)}")