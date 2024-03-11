# Streamlit Imports
import streamlit as st

# Langchain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.prompts import(
    PromptTemplate, ChatPromptTemplate, 
    MessagesPlaceholder, SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# Import system
import sys
import os

# Import other modules
from config import Config
import pickle
import random
import time
import re

# ------------------------------------------------------ FUNCTIONS ------------------------------------------------------ #
# Function to get the text chunks from the PDFs
def get_document_text_chunks():

    # Initialize the text chunks list
    docs_text_chunks = []

    # Retrive all the PDF files from the temp_pdf_store folder. Output of file_list = ['file1.pdf', 'file2.pdf']
    files = filter(lambda f: f.lower().endswith(".pdf"), os.listdir("temp_pdf_store"))
    file_list = list(files)

    # Loop through the PDF files and extract the text chunks
    for file in file_list:
        
        # Retrieve the PDF file
        loader = PyPDFLoader(os.path.join('temp_pdf_store', file)) # f"{os.getcwd()}\\temp_pdf_store\\{file}"

        # Get the text chunks of the PDF file, accumulate to the text_chunks list variable becaus load_and_split() returns a list of Document
        docs_text_chunks += loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 80,
            length_function = len,
            separators= ["\n\n", "\n", " ", ""]
        ))
    return docs_text_chunks

# Function to get embeddings
def get_vectors_embedding(docs_text_chunks):
    if Config.LLMCONFIG.get('CHOSEN_LLM') == "Azure OpenAI":
        embeddings = AzureOpenAIEmbeddings(
            deployment = Config.LLMCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2023-05-15", 
            openai_api_type = st.secrets["OPENAI_API_TYPE"], 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model=Config.LLMCONFIG.get('EMBEDDMODEL'), task_type="retrieval_query", google_api_key=st.secrets["GOOGLE_API_KEY"])
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        embeddings = AzureOpenAIEmbeddings(
            deployment = Config.LLMCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2023-05-15", 
            openai_api_type = st.secrets["OPENAI_API_TYPE"], 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")

    vector_embeddings = FAISS.from_documents(documents=docs_text_chunks, embedding=embeddings)
    return vector_embeddings

# Function to create the LLM chat agent
def get_conversation_chain(vector_embeddings):
    if Config.LLMCONFIG.get('CHOSEN_LLM') == "Azure OpenAI":
        llm = AzureChatOpenAI(
            deployment_name = "my-dna-gpt35turbo", 
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2023-05-15", 
            openai_api_type = st.secrets["OPENAI_API_TYPE"], 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"],
            max_tokens = 550,
            model_kwargs={"top_p": 0.9}
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        llm = GoogleGenerativeAI(model=st.session_state.gemini_pro_model, google_api_key=st.secrets["GOOGLE_API_KEY"])    
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        llm = ChatGroq(temperature=0, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")
    
    custom_template = """You are a powerful AI Assistant. Given the following conversation and a 
    follow up question, rephrase the follow up question to be a standalone question. 
    At the end of standalone question add this 'Answer the question in English language.' 
    If you do not know the answer reply with 'I am sorry, I dont have enough information'.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    
    conversation_chain = ConversationalRetrievalChain.from_llm( # from_chain_type
        llm=llm, # Use the llm to generate the response, we can use better llm such as GPT-4 model from OpenAI to guarantee the quality of the response. For exp, the resopnse is more human-like
        retriever=vector_embeddings.as_retriever(
            search_type = Config.VECTORSTORECONFIG.get("SEARCH_TYPE"),
            search_kwargs = Config.VECTORSTORECONFIG.get("SEARCH_KWARGS"),
        ),
        return_source_documents=True,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        condense_question_llm=llm # Can use cheaper and faster model for the simpler task like condensing the current question and the chat history into a standalone question with GPT-3.5 if you are on budget. Otherwise, use the same model as the llm
    )
    return conversation_chain

# Function to save the uploaded file to the local temp_pdf_store folder
def save_uploadedfile(uploadedfile):

    with open(os.path.join("temp_pdf_store", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

def initialize_session_state():
    # Set a default model
    if "gemini_pro_model" not in st.session_state:
        st.session_state["gemini_pro_model"] = "gemini-pro"
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = None
    if "is_vector_embeddings" not in st.session_state:
        st.session_state.is_vector_embeddings = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def reset_session_state():    
    # Delete all the keys in session state
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Initialize the default session state variables again
    initialize_session_state()

# Function to remove files in the Upload folder
def remove_files():
    path = os.path.join(os.getcwd(), 'temp_pdf_store')
    for file_name in os.listdir(path):
        # construct full file path
        # file = path + '\\' + file_name
        file = os.path.join(path, file_name)
        if os.path.isfile(file) and file.endswith(".pdf"): # Only remove the PDF files
            print('Deleting file:', file)
            os.remove(file)