from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import urllib.parse
import urllib.request
from operator import itemgetter

# import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import tool
load_dotenv()

#store key
#st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY") # Get the api key from .env file and store it in a session variable
#if not st.session_state['openai_api_key']: # Check the api key exists
    #st.error("OpenAI API key not found")
    #st.stop()

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

start_date = '2024-03-24'
end_date = '2024-03-24'


def get_events(start_date: str, end_date: str, human_input: str) -> str:
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

print("Hello! Please let us know what kind of events you're looking for.\n Enter 'exit' to quit.")

human_input = ""
while human_input != "exit":
    human_input = input("User: ")
    
    print(get_events(start_date,end_date,human_input))
        

    # result = agent.invoke({"input": human_input})
    # print("Helper: ", result['output'])
exit()