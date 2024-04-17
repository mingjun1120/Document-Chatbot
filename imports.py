# Streamlit Imports
import streamlit as st
from streamlit.web import cli as stcli
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Langchain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.chains.conversational_retrieval.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import (
    PromptTemplate, ChatPromptTemplate, 
    MessagesPlaceholder, SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from chromadb.errors import InvalidDimensionException

# Import env variables
from dotenv import load_dotenv

# Import system
import sys
import os

# Import other modules
from config import Config
from functions import *
import time
import re

# Import Spire modules
from spire.doc import Document, FileFormat
from spire.doc.common import *