
import pandas as pd
from typing import Annotated, Literal, TypedDict

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

from langchain_chroma import Chroma

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

@tool
def tool_get_dataset(query: str, US_Canada_CPI_data: pd.DataFrame, path: str = '../../data/processed/USA_CPI_Processed_2018_2019.csv', US_or_Canada_dataset: str = "USA") -> str:
    if US_or_Canada_dataset == "USA":
        df = pd.read_csv(path)
    else:
        path = '../../data/processed/Canada_CPI_Processed_2018_2019.csv'
        df = pd.read_csv(path)
    
@tool
def load_website_data(website: Annotated[str, Predicate(lambda x: re.search('www.', x))])->list:
    "This tool loads a website using WebBaseLoader and returns a list of documents about the website"
    load_web = WebBaseLoader(website)
    return load_web.load()[0].page_content

@tool
def get_news_data_from_website(query, website_content_docs, llm):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(website_content_docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm
    return rag_chain.invoke({"question": query})

def add_tools(all_tools: list = [tool_get_dataset, load_website_data, get_news_data_from_website]):
    
    return ToolNode(all_tools)
# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

# Define the final model
def final_model(access_key_id: str, secret_access_key: str, all_tools: list = [tool_get_dataset, load_website_data, get_news_data_from_website]):
    # Define the model
    
    model = ChatBedrockConverse(
        model_name='meta.llama3-1-70b-instruct-v1:0', region_name='us-west-2',
        max_tokens=100,
        temperature=0.5,
        max_turns=1,
        max_tokens_per_turn=100,
        max_turns_per_message=1,
        max_messages=1,
        max_tokens_per_message=100,
        max_tokens_per_conversation=100,
        max_conversations=1
    ).bind_tools(all_tools)
    return model

# Define the function that calls the model
def call_model(state: MessagesState, model):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def work_flow(query, tool_nodes = add_tools()):
    # Define a new graph
    workflow = StateGraph(MessagesState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_nodes)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.add_edge(START, "agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", 'agent')



    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    #compile the workflow
    app = workflow.compile(checkpointer=checkpointer)
    system = """system: You are a helpful AI that answers questions about the news. """
    # Use the Runnable
    final_state = app.invoke(
        {"messages": [HumanMessage(content=system + "question: " + query)]},
        config={"configurable": {"thread_id": 42}}
    )

    return final_state["messages"][-1].content