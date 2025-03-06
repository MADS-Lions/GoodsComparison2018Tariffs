
import pandas as pd
import re
from typing import Annotated, Literal, TypedDict
import boto3
from langchain.tools import tool

from langchain_aws import ChatBedrockConverse

from annotated_types import Optional, Annotated, Predicate

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from langchain_core.messages import AIMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOllama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from operator import add
import json
import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from typing import Union
from langchain_core.messages import BaseMessage

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

import os
from textwrap import TextWrapper
os.environ.get('AWS_ACCESS_KEY_ID', input())
os.environ.get('AWS_SECRET_ACCESS_KEY', input())
os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')


@tool
def tool_get_dataset(question: str = '', path: str = '../data/processed/USA_CPI_Processed_2018_2019.csv', US_or_Canada_dataset: str = "USA") -> str:
    """This tool loads the dataset about either the American or Canadian CPI goods and services pandas dataframe
    and does manipulation on the dataset to try to answer questions it takes in all STRING values for 3 of the following Arguments
    
    Accepted Arguments:
    question: str - The question should inputed here based on the question asked by the user so the tool can help answer the question by exploring the pandas dataframe
    
    US_or_Canada_dataset: str - which is either 'Canada' or 'USA' and is the dataset to be loaded

    path: str - which is the path to the dataset which is already given

    Returns:
    Returns a string with the dataset loaded and any analysis that you feel is necessary to answer the question
    
    """
    print(question)
    
    if US_or_Canada_dataset == "USA":
        df = pd.read_csv(path)
    else:
        path = '../data/processed/Canada_CPI_Processed_2018_2019.csv'
        df = pd.read_csv(path)
    
    # Perform some basic analysis based on the query some examples of what you could do
    result = f"Dataset loaded for {US_or_Canada_dataset}. "
    result += f"Shape: {df.shape}. "
    result += f"Columns: {', '.join(df.columns)}. "
    result += f"Sample data: {df.head().to_dict()}"
    
    
web_list = ["https://www.piie.com/blogs/trade-and-investment-policy-watch/2019/trumps-fall-2019-china-tariff-plan-five-things-you",
            "https://www.cfr.org/article/what-trumps-trade-war-would-mean-nine-charts",
            "https://www.brookings.edu/research/the-us-china-trade-war-a-timeline/",
            "https://www.rba.gov.au/information/foi/disclosure-log/pdf/232431.pdf",
            "https://www.federalreserve.gov/econres/feds/files/2019086pap.pdf",
            "https://secure.caes.uga.edu/extension/publications/files/pdf/C%201259_1.PDF",
            "https://apps.fas.usda.gov/newgainapi/api/Report/DownloadReportByFileName?fileName=Chi[…]oducts_Beijing_China%20-%20Peoples%20Republic%20of_5-17-2019",
            "https://www.piie.com/sites/default/files/documents/trump-trade-war-timeline.pdf",
            "https://www.retailcouncil.org/wp-content/uploads/2018/08/RCC_NAFTA_Comments_June_15_2018_FINAL-1.pdf"]
            
@tool
def load_website_data(question: str, website: Annotated[str, Predicate(lambda x: re.search('www.', x))]) -> dict:
    """This tool loads a website using WebBaseLoader and returns structured data from the website
    Arguments:
    
    Accepts question: which is the question to be answered using the website content
    Accepts website: str which is the url of the website to be loaded

    Returns:
    Returns a dictionary with the following with the url, content of the website, the word count, and the original question
    """
    print(question)
    load_web = WebBaseLoader(website)
    content = load_web.load()[0].page_content
    
    return {
        "url": website,
        "content": content,
        "word_count": len(content.split()),
        "question": question
    }

@tool
def choose_website_from_list(question: str, website_list: List[str]) -> str:
    """This tool is to choose a website from a list of websites to load and answer the question
    Accepts question: str which is the question to be answered using the website content
    Accepts website_list: List[str] which is a list of websites to choose from to load

    Returns:
    Returns a string with the answer to the question based on the content from the website
    """
    print(question)
    for i, website in enumerate(website_list):
        print(f"{i+1}. {website}")
        load_web = WebBaseLoader(website)
        content = load_web.load()[0].page_content
        print(content)
    choice = int(input("Choose a website to load: "))
    return f"Based on the content from {website}, here's the answer: {website}"

@tool
def get_news_data_from_website(question: str, website_data: dict) -> str:
    """This tool uses the query and the website_data from load_website_data to get the news data from the website and answer the query
    Accepts question: str which is the question to be answered using the website content
    Accepts website_data: dict which is the structured data from the website
    Accepts llm: OllamaLLM which is the model to be used to answer the question

    Returns:
    Returns a string with the answer to the question based on the content from the website
    """
    
    
    print(question)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(website_data["content"])
    
    vectorstore = Chroma.from_texts(texts=splits, embedding=HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_tools
    result = rag_chain.invoke({"question": question})
    
    return f"Based on the content from {website_data['url']} ({website_data['word_count']} words), here's the answer: {result}"



def final_model():
    """This function is to make an agent for the user to interact with the tools and the LLM model to answer questions"""
    

    
    
    llm = ChatBedrockConverse(model='meta.llama3-1-70b-instruct-v1:0')
    

    llm_tools = llm.bind_tools([tool_get_dataset, load_website_data, get_news_data_from_website])

    return llm_tools



def add_tools(all_tools: list = [tool_get_dataset, load_website_data, get_news_data_from_website]):
    return ToolNode(all_tools)



def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


    

llm_tools = final_model()






def run_agent(state: MessagesState):
    query = state['messages']
    response = llm_tools.invoke(query)
    return {"messages": query + [response]}





def work_flow(query, max_iterations=8):
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("agent", run_agent)
    workflow.add_node("tools", add_tools())
    
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        
    )
    workflow.add_edge("tools", "agent")
    
    checkpointer = MemorySaver()
    
    app = workflow.compile(checkpointer=checkpointer)
    memory = {}
    system = "System: You are a helpful AI that answers questions about the news related to the 2018 tariffs that impacted inflation of goods and services in Canada and USA. Use the tools at your disposal to look up news articles or data from the CPI goods and services for Canada and US as a surrogate inflation to help support your answer."
    
    initial_message = HumanMessage(content=f"{system}\n\nQuestion: {query}")
    print("Initial message:", initial_message)

    response = app.invoke(
        {"messages": [initial_message]},
        config={"configurable": {"thread_id": 42}, "recursion_limit": max_iterations},
    )
    
    wrapper = TextWrapper(width=80)
    # Process the final state to extract the answer
    
    messages = response['messages'][-1]
    wrap_text = wrapper.wrap(messages.content)
    return wrap_text





    
    
    
