
# from langchain_community.chat_models import ChatOpenAI
# from dotenv import load_dotenv
# import os
# import json
# import urllib.parse
# import urllib.request
# from operator import itemgetter

# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.tools import tool
# load_dotenv()

# # Store OpenAI API key
# st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY") # Get the api key from .env file and store it in a session variable
# if not st.session_state['openai_api_key']: # Check the api key exists
#     st.error("OpenAI API key not found")
#     st.stop()
# model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# # Title
# st.title("Find Events AT Michigan")

# # Initialize chat history
# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = ["Hello! Please let us know what kind of events you're looking for."]

# if 'past' not in st.session_state:
#     st.session_state['past'] = ["Hey"]

# # Initialize a LangChain chat agent
# #llm = ChatOpenAI(temperature=0)
# tools = []
# #prompt = hub.pull("hwchase17/openai-tools-agent")

# #agent = create_openai_tools_agent(llm, tools, prompt)
# #agent = create_openai_tools_agent(model, tools, prompt)

# from langchain.memory import ChatMessageHistory

# history = []

# start_date = '2024-03-24'
# end_date = '2024-03-24'
# #time selection
# start_date = str(st.sidebar.text_input('start_date'))
# end_date = str(st.sidebar.text_input('end_date'))

# url = f'http://events.umich.edu/list/json?filter=all&range={start_date}to{end_date}&v=2'

# def get_events(start_date: str, end_date: str, human_input: str) -> str:
#     url = f'http://events.umich.edu/list/json?filter=all&range={start_date}to{end_date}&v=2'

#     with urllib.request.urlopen(url) as response:
#         data = response.read()

#     events_json = json.loads(data)

#     event_descriptions = []
#     for event in events_json:
#         if event["combined_title"] == None:
#             event["combined_title"] = "An event"
#         if event["location_name"] == None:
#             event["location_name"] = " an unknown location"
        
#         if event["room"] != "":
#             new_event = (event["combined_title"] + " is happening in " +  event["location_name"] + " " + event["room"])
#         else: 
#             new_event = (event["combined_title"] + " is happening in " +  event["location_name"])

#         if event["date_start"] == event["date_end"]:
#             new_event += ". The date of the event is " + event["date_start"] + "."
#         else:
#             new_event += ". The dates of the event are from " + event["date_start"] + " to " + event["date_end"] + "."
        
#         if event["time_start"] == "00:00:00" and event["time_end"] == "23:59:59":
#             new_event += "The event is happening all day."
#         else:
#             new_event += ". The time the event is happening is from " + event["time_start"] + " to " + event["time_end"] + "."
        

#         new_event += ". A description of the event is: " + event["description"]

#         if event["cost"] != "":
#             new_event += ". Cost information is: " + event["cost"]

#         event_descriptions.append(new_event)
        
#     vectorstore = FAISS.from_texts(event_descriptions, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

#     retriever = vectorstore.as_retriever(k=5)

#     template = """Provide information about events the user might be interested in, including location, date, and time information about those events:
#     {context}

#     Question: {question}
#     """

#     model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

#     chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | PromptTemplate.from_template(template)
#         | model
#         | StrOutputParser()
#     )   

#     return chain.invoke(human_input)

# print("Hello! Please let us know what kind of events you're looking for.\n Enter 'exit' to quit.")

# human_input = ""
# while human_input != "exit":
#     human_input = input("User: ")
    
#     print(get_events(start_date,end_date,human_input))
        

#     # result = agent.invoke({"input": human_input})
#     # print("Helper: ", result['output'])
# exit()


# To run this example, make sure you run this in command line:
# pip install -r requirements.txt
# Then run the streamlit with this command:
# streamlit run streamlit-example.py
# (make sure you get the path to this .py file correct)

import os
import shutil
import streamlit as st
import tempfile
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.tools import tool
from datetime import date
import datetime
from daterangeparser import parse
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import urllib.parse
import urllib.request
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# Store OpenAI API key
st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY") # Get the api key from .env file and store it in a session variable
if not st.session_state['openai_api_key']: # Check the api key exists
    st.error("OpenAI API key not found")
    st.stop()

@tool
def get_events(human_input: str) -> str:
    """Function that finds events within the provided date range and outputs events the user might be interested in as a list."""
    url = f'http://events.umich.edu/list/json?filter=all&range={start_date}to{end_date}&v=2'

    with urllib.request.urlopen(url) as response:
        data = response.read()

    events_json = json.loads(data)

    event_descriptions = []
    for event in events_json:
        if event["combined_title"] == None:
            event["combined_title"] = "An event"
        if event["location_name"] == None:
            event["location_name"] = " an unknown location"
        
        if event["room"] != "":
            new_event = (event["combined_title"] + " is happening in " +  event["location_name"] + " " + event["room"])
        else: 
            new_event = (event["combined_title"] + " is happening in " +  event["location_name"])

        if event["date_start"] == event["date_end"]:
            new_event += ". The date of the event is " + event["date_start"] + "."
        else:
            new_event += ". The dates of the event are from " + event["date_start"] + " to " + event["date_end"] + "."
        
        if event["time_start"] == "00:00:00" and event["time_end"] == "23:59:59":
            new_event += "The event is happening all day."
        else:
            new_event += ". The time the event is happening is from " + event["time_start"] + " to " + event["time_end"] + "."
        

        new_event += ". A description of the event is: " + event["description"]

        if event["cost"] != "":
            new_event += ". Cost information is: " + event["cost"]

        event_descriptions.append(new_event)
        
    vectorstore = FAISS.from_texts(event_descriptions, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    retriever = vectorstore.as_retriever(k=5)

    template = """Provide information about events the user might be interested in, including location, date, and time information about those events:
    {context}

    Question: {question}
    """

    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(template)
        | model
        | StrOutputParser()
    )   

    return chain.invoke(human_input)

# Title
st.title("Find Events AT Michigan")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! What kind of events are you looking for?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hi"]


tools = [get_events]
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
system_message = f"""
    You are a chatbot helper that helps users find events happening around Ann Arbor that they are interested in going to.

    You must ask them for the dates they're interested in going to and the kinds of events they are interested in. 
    You must then suggest events happening in the given date range that they would be interested in. 

    You have several tools at your disposal, which you will have to take advantage of to figure out when the user is interested in going to an event and what kinds of events they like.
    - You can leverage a function for finding events for the user, given user input about what kinds of events they are looking for.
    - You can ask the user for more information, such as what exact kinds of events they are interested in.
    """

from langchain.agents import initialize_agent

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=model,
    agent_kwargs={
        "system_message": system_message,
    },
    verbose=True,
    max_iterations=3
)

response_container = st.container()
container = st.container()


#time selection
start_date = st.sidebar.date_input('Start date', date.today())
end_date = st.sidebar.date_input('End date', date.today() + datetime.timedelta(days=7))


# Create streamlit containers for chat history and user input
# Display user input area
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Input", placeholder="Enter a message", 
                                   key='input', label_visibility="collapsed")
        submit_button = st.form_submit_button(label='Send')

    # This runs when user enters message to chat bot
    if submit_button and user_input:
        # Get a response from the llm, by giving the user message and chat history to the agent
        result = agent.invoke({"input": user_input, "chat_history": st.session_state['history']})

        # Add the user message and llm response to the chat history
        st.session_state['history'].append(HumanMessage(content=user_input))
        st.session_state['history'].append(AIMessage(result["output"]))

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result["output"])

# Dislplay chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")



# st.write('You selected period from', start_date, 'to', end_date)
