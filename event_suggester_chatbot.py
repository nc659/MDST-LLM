import os
import shutil
import dateparser
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

#time selection
st.session_state['start_date'] =  date.today()
st.session_state['end_date'] = date.today() + datetime.timedelta(days=7)
# st.session_state['start_date'] = st.sidebar.date_input('Start date', st.session_state['start_date'] )
# st.session_state['end_date'] = st.sidebar.date_input('End date', st.session_state['end_date'])

st.sidebar.write('Date range for events:')

@tool
def get_events(human_input: str) -> str:
    """Function that finds events within the provided date range and outputs events the user might be interested in as a detailed, bullet-point list."""
    url = f'http://events.umich.edu/list/json?filter=all&range={st.session_state['start_date']}to{st.session_state['end_date']}&v=2'
    print("dates::: ", st.session_state['start_date'], st.session_state['end_date'])
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

    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(template)
        | model
        | StrOutputParser()
    )   

    return chain.invoke(human_input)

@tool
def set_start_date(input_expr: str) -> int:
    """Function that, given date information in weekday or date formats about when a user is interested in going to an event, sets the start date in year-month-day format. Do not include 'this' in input string."""
   
    date = dateparser.parse(input_expr, settings={'PREFER_DATES_FROM': 'future'})
    if (date is not None):
        st.session_state['start_date'] = date.date()
        st.sidebar.write('Start Date: ' + str(st.session_state['start_date']) )

        return 1
    else:
        return 0

@tool
def set_end_date(input_expr: str) -> int:
    """Function that, given date information in weekday or date formats about when a user is interested in going to an event, sets the start date in year-month-day format. Do not include 'this' in input string."""
   
    date = dateparser.parse(input_expr, settings={'PREFER_DATES_FROM': 'future'})
    if (date is not None):
        st.session_state['end_date'] = date.date()
        st.sidebar.write('End Date: ' + str(st.session_state['end_date']) )
        return 1
    else:
        return 0


# Title
st.title("Find Events AT Michigan")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! What kind of events are you looking for?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hi"]


tools = [get_events, set_start_date, set_end_date]
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
system_message = f"""
    You are a chatbot helper that helps users find events happening around Ann Arbor that they are interested in going to.

    You must ask them for the dates they're interested in going to and the kinds of events they are interested in. 
    You must then suggest events happening in the given date range that they would be interested in. 

    You have several tools at your disposal, which you will have to take advantage of to figure out when the user is interested in going to an event and what kinds of events they like.
    - In order to set the date range required to figure out what events are available, you will need to set start and end dates using set_start_date and set_end_dates. If the date range is one day, call both functions and set them to be the same day.
        - If the return values of the set_date functions are 0, ask the user to provide more specific date information
    - You can leverage a function for finding events for the user, given user input about what kinds of events they are looking for. You need user input about dates they're interested in before calling this function.
    - You can ask the user for more information, such as what exact kinds of events they are interested in and dates they are interested in.
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
    max_iterations=5
)

response_container = st.container()
container = st.container()



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
