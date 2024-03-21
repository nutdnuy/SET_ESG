# Import necessary libraries
from openai._client import OpenAI
import streamlit as st
import requests
import time
import json
import requests
import os
import sys
import pandas as pd
from pdfminer.high_level import extract_text

sys.path.append('/')
load_dotenv()

# use dotenv to load environment variables
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assistant_id =  os.getenv("assistant_API_KEY")
client = OpenAI(api_key= api_key )
st.session_state.start_chat = False
# Initialize session state variables
if "file_id_list" not in st.session_state:
    st.session_state.file_id_list = []

# Set up the Streamlit page with a title and icon
st.set_page_config(page_title="Quant Corner ESG", page_icon=":speech_balloon:")

# Create a sidebar for API key configuration and additional features
st.sidebar.header("Configuration")
# input box for openai api key
# api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
# fmp_api_key = st.sidebar.text_input("Enter your Financial Modeling Prep API key", type="password")
# st.sidebar.markdown(
#     """
#     You can get an API key from [OpenAI](https://platform.openai.com/signup) and [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
#     """
# )
# show image
st.sidebar.image("QR_phatra.jpg", use_column_width=True)
P0 = st.sidebar.text_input("Who are you?")
file = st.sidebar.file_uploader("UP load ESG one report", accept_multiple_files=False)


    






if api_key:
    OpenAI.api_key = api_key
    client = OpenAI(api_key=api_key)
    st.session_state.start_chat = True
    st.session_state.msgs = []



if "trigger_assistant" not in st.session_state:
    st.session_state.trigger_assistant = False


def summarise_agent (text ) :
    from pdfminer.high_level import extract_text
    # The variable 'sys_prompt' should contain the message text
    user_message = "Please summarise: " + text[:10000]
    run1 = submit_message(assistant_id_sum, thread_id_sum , user_message)
    run1 = wait_on_run(run1, thread_id_sum)
    a = get_response(thread_id_sum)
    dic = {}
    for m in a:
        #print(f"{m.role}: {m.content[0].text.value}")
        dic[m.role] = m.content[0].text.value
            
    lst_ .append(dic['assistant'])
    str =  ",".join(str(element) for element in lst_ )
    return str


def get_E_data(symbol):
    try :
        df =pd.read_excel("data/SET_E.xlsx")
        filtered_data = df[df['Symbol'] == symbol].iloc[-1].to_json()
    except : filtered_data =[]
    return filtered_data

def get_S_data(symbol):
    try :
        df =pd.read_excel("data/SET_S.xlsx")
        filtered_data = df[df['Symbol'] == symbol ].iloc[-1].to_json()
    except :filtered_data =[]
    return filtered_data

def get_G_data(symbol):
    try :
        df =pd.read_excel("data/SET_G.xlsx")
        filtered_data = df[df['Symbol'] == symbol ].iloc[-1].to_json()
    except :filtered_data =[]
    return filtered_data

def summarise_agent (text) :
    from pdfminer.high_level import extract_text
    # The variable 'sys_prompt' should contain the message text
    user_message = "Please summarise: " + text[:30000]
    run1 = submit_message(assistant_id_sum, thread_id_sum , user_message)
    run1 = wait_on_run(run1, thread_id_sum)
    a = get_response(thread_id_sum)
    dic = {}
    for m in a:
        #print(f"{m.role}: {m.content[0].text.value}")
        dic[m.role] = m.content[0].text.value
            
    lst_ .append(dic['assistant'])
    str =  ",".join(str(element) for element in lst_ )
    return str



# Define the function to process messages with citations
def process_message_with_citations(message):
    """Extract content and annotations from the message and format citations as footnotes."""
    #  handle MessageContentImageFile
    if message.content[0].type == "image":
        return f"![{message.content[0].filename}]({message.content[0].url})"
    else:
        message_content = message.content[0].text
    annotations = message_content.annotations if hasattr(message_content, 'annotations') else []
    citations = []

    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(annotation.text, f' [{index + 1}]')

        # Gather citations based on annotation attributes
        if (file_citation := getattr(annotation, 'file_citation', None)):
            # Retrieve the cited file details (dummy response here since we can't call OpenAI)
            cited_file = {'filename': 'cited_document.pdf'}  # This should be replaced with actual file retrieval
            citations.append(f'[{index + 1}] {file_citation.quote} from {cited_file["filename"]}')
        elif (file_path := getattr(annotation, 'file_path', None)):
            # Placeholder for file download citation
            cited_file = {'filename': 'downloaded_document.pdf'}  # This should be replaced with actual file retrieval
            citations.append(f'[{index + 1}] Click [here](#) to download {cited_file["filename"]}')  # The download link should be replaced with the actual download path

    # Add footnotes to the end of the message content
    full_response = message_content.value + '\n\n' + '\n'.join(citations)
    return full_response


if "thread_id" not in st.session_state:
    assistant = client.beta.assistants.retrieve(assistant_id)
    st.session_state.assistant_instructions = assistant.instructions
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    print(f'Thread ID: {thread.id}')


st.session_state.start_chat =  True
#st.sidebar.image("QR_phatra.png", use_column_width=True)












if "trigger_assistant" not in st.session_state:
    st.session_state.trigger_assistant = False

if "msgs" not in st.session_state:
    st.session_state.msgs = []















# Main chat interface setup
st.title("ESG Materiality Assessment analysis")
st.write("Upload one report GPT will provide  Materiality  ")

if st.session_state.msgs == []:
    st.markdown("""
        Hi, I'm QC ESG Materiality Assessment analysis Advisor. Provide me with the symbol and one report
        """)



    # Display existing messages in the chat
    for message in st.session_state.msgs:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for the user
    prompt = st.chat_input("Please ask question about esg ", key="chat_input")
    if file : 
        text = extract_text("y2023-sd-en.pdf")


    
    if prompt or st.session_state.trigger_assistant:
        
        if st.session_state.trigger_assistant:
            print("trigger assistant")
            prompt = st.session_state.trigger_assistant
            st.session_state.trigger_assistant = False

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.msgs.append({"role": "user", "content": prompt})
        msgs = st.session_state.msgs

        with st.spinner("Thinking..."):
            thread_id = st.session_state.thread_id
            # Add the user's message to the existing thread
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )

            # Create a run with additional instructions
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                #this instruction will overwrite the instruction in the assistant
                # instructions=st.session_state.assistant_instructions + "\n\n" + "", 
            )

            # Poll for the run to complete and retrieve the assistant's messages

            while run.status != 'completed':
                run = client.beta.threads.runs.retrieve(
                    thread_id=st.session_state.thread_id,
                    run_id=run.id
                )
                time.sleep(1)
                if run.status == "requires_action":
                    tools_output = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        f = tool_call.function
                        print(f)
                        f_name = f.name
                        f_args = json.loads(f.arguments)

                        print(f"Launching function {f_name} with args {f_args}")
                        tool_result = eval(f_name)(**f_args)
                        tools_output.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": str(tool_result),
                            }
                        )
                    print(f"Will submit {tools_output}")
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=st.session_state.thread_id,
                        run_id=run.id,
                        tool_outputs=tools_output,
                    )
                if run.status == "completed":
                    print(f"Run status: {run.status}")
                    
                if run.status == "failed":
                    print("Abort")
                    #print the error message
                    print(run.last_error)

            # Retrieve messages added by the assistant
            messages = client.beta.threads.messages.list(
                thread_id=st.session_state.thread_id
            )

            # Process and display assistant messages
            assistant_messages_for_run = [
                message for message in messages 
                if message.run_id == run.id and message.role == "assistant"
            ]  
            for message in assistant_messages_for_run:
                full_response = process_message_with_citations(message)
                st.session_state.msgs.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    st.markdown(full_response, unsafe_allow_html=True)
