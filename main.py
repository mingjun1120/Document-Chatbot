# Streamlit Imports
import streamlit as st
from streamlit.web import cli as stcli
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Langchain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# Import env variables
from dotenv import load_dotenv

# Import system
import sys
import os

# Import other modules
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
            chunk_size = 750,
            chunk_overlap = 80,
            length_function = len,
            separators= ["\n\n", "\n", ".", " "]
        ))

    return docs_text_chunks

def get_vectors_embedding(docs_text_chunks): 
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query", google_api_key=st.secrets["GOOGLE_API_KEY"])
    embeddings = AzureOpenAIEmbeddings(deployment = "text-embedding-ada-002", 
        openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
        openai_api_version = "2023-05-15", 
        openai_api_type = st.secrets["OPENAI_API_TYPE"], 
        azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
    )
    vector_embeddings = FAISS.from_documents(documents=docs_text_chunks, embedding=embeddings)

    return vector_embeddings

def get_conversation_chain(vector_embeddings):

     llm = GoogleGenerativeAI(model=st.session_state.gemini_pro_model, google_api_key=api_key, temperature=0.5)
     llm = AzureChatOpenAI(deployment_name = "my-dna-gpt35turbo", 
         openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
         openai_api_version = "2023-05-15", 
         openai_api_type = st.secrets["OPENAI_API_TYPE"], 
         azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"])
 
     
   # llm = AzureOpenAI(deployment_name = "my-dna-gpt35turbo", 
   #     # model_name = "gpt-3.5-turbo"
   #     openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
   #     openai_api_version = "2023-05-15", 
   #     openai_api_type = st.secrets["OPENAI_API_TYPE"], 
   #     azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"],
   #     temperature=0.5,
   #     top_p=0.5
   # )
   # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm( # from_chain_type
        llm=llm, # Use the llm to generate the response, we can use better llm such as GPT-4 model from OpenAI to guarantee the quality of the response. For exp, the resopnse is more human-like
        retriever=vector_embeddings.as_retriever(),
        condense_question_prompt=condense_ques_prompt,
        # memory=memory,
        return_source_documents=True,
        chain_type="map_reduce",
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
    # Use to clear the inputs in st.text_input for URL 1, URL 2, URL 3 when the user clicks the "Reset All" button
    # st.session_state.URL_4, st.session_state.URL_5, st.session_state.URL_6 = st.session_state.URL_1, st.session_state.URL_2, st.session_state.URL_3
    # st.session_state.URL_1, st.session_state.URL_2, st.session_state.URL_3 = '', '', ''
    
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

# ------------------------------------------------------ GLOBAL VARIABLES ------------------------------------------------------ #

# Load the environment variables
load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# google_key = st.secrets["GOOGLE_API_KEY"]
# azure_key = st.secrets["AZURE_OPENAI_API_KEY"]
# azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
# azure_type = st.secrets["OPENAI_API_TYPE"]

prompt_template = """Given the following conversation and a follow-up question from the user, rephrase the follow-up question to be a standalone question, in its original language. Subsequently, understand the context of the question and retrieve relevant information on the standalone question from the VectorStoreRetriever to generate an answer. If you can't find relevant information to asnwer the question, explicitly state that no relevant references were found.

Chat History:
{chat_history}
Follow-Up User Input: {question}
Standalone Question:"""
condense_ques_prompt = PromptTemplate.from_template(prompt_template)

# Set the tab's title, icon and CSS style
page_icon = ":speech_balloon:"  # https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="PDF Chat App", page_icon=page_icon, layout="centered")

# Page header
# st.header(body=f"Document :books: ChatBot {page_icon}")
st.header(body=f"DocEYE :books: ChatBot {page_icon}")

def main():

    # Initialize the session state variables
    initialize_session_state()

    # ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #
    # Sidebar contents
    with st.sidebar:

        # Upload PDF Files
        st.subheader("Your documents")
        docs = st.file_uploader(label="Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True, type=["pdf"], 
            key=st.session_state["file_uploader_key"]
        )
        
        # Process documents
        process_button = st.button(label="Process docs")

        if process_button:
            if docs != []:
                # st.session_state.clear() # Clear the session state variables
                with st.spinner(text="Data Loading...✅✅✅"):
                    
                    # Save the uploaded files (PDFs) to the "temp_pdf_store" folder
                    for doc in docs:
                        save_uploadedfile(doc)

                with st.spinner(text="Text Splitting...✅✅✅"):
                    # Get the text chunks from the PDFs
                    docs_text_chunks = get_document_text_chunks()

                with st.spinner(text="Building Embedding Vector...✅✅✅"):
                    # Create Vector Store
                    vector_embeddings = get_vectors_embedding(docs_text_chunks)
                    st.session_state.is_vector_embeddings = True
                    
                    # Remove the PDFs from the Upload folder
                    remove_files()

                with st.spinner(text="Building Conversation Chain...✅✅✅"):
                    # Create conversation chain
                    st.session_state.conversation_chain = get_conversation_chain(vector_embeddings)
                
                # Print System Message at the end
                st.success(body=f"Done processing!", icon="✅")

            # Use to check if URLs are processed. If not processed, users will be asked to upload PDFs when ask questions.
            st.session_state.is_processed = process_button

        if docs != []:
            st.session_state.docs = docs
        else:
            st.session_state.docs = None
            st.session_state.is_vector_embeddings = False

        add_vertical_space(num_lines=1)

        # Web App References
        st.markdown('''
        ### About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Gemini Pro](https://ai.google.dev/)
        ''')
        st.write("Made ❤️ by Lim Ming Jun")

        # Reset button part
        reset = st.button('Reset All', on_click=reset_session_state)
        if reset:
            # for key in st.session_state.keys():
            #     del st.session_state[key]
            # initialize_session_state()
            st.rerun()
    
    # ------------------------------------------------------ MAIN LAYOUT------------------------------------------------------ #
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Enter your query:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            error_message = "Sorry, please upload your document(s) and click the **Process docs** button before querying!"
            if prompt != None and prompt.isspace() == False:
                if st.session_state.docs != None and st.session_state.is_processed != None and st.session_state.is_vector_embeddings == True:
                    # result will be a dictionary of this format --> {"answer": "", "sources": ""}
                    result = st.session_state.conversation_chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
                    st.session_state.chat_history.append((prompt, result.get("answer")))
                    assistant_response = result.get("answer")
                    
                    # Get the refeence source
                    references = ["**Sources:**"]
                    if result.get("source_documents") != []:
                        for index, doc in enumerate(result.get("source_documents")):
                            references.append(f"{index + 1}. " + os.path.split(doc.metadata.get('source'))[1] + " - Page " + str(doc.metadata.get('page')))
                    
                    # Simulate stream of response with milliseconds delay
                    for index, chunk in enumerate(assistant_response.split() + references):
                        
                        if chunk == '**Sources:**':
                            full_response += f'\n\n{chunk}\n'
                        elif bool(re.match(r'^\d+\.', chunk)):
                            full_response += f'{chunk}\n'
                        else:
                            full_response += chunk + " "
                        
                        time.sleep(0.05)
                        
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                
                else:
                    assistant_response = error_message
                    # Simulate stream of response with milliseconds delay
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            
                # # Add assistant response to chat history
                # st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                if prompt.isspace():
                    prompt = None
                
                if st.session_state.messages != [] and st.session_state.messages[-1]["content"] == error_message and prompt == None:
                    # Simulate stream of response with milliseconds delay
                    for index, chunk in enumerate(assistant_response.split()):
                        full_response += chunk + " " 
                        time.sleep(0.05)
                        
                        # Add a blinking cursor to simulate typing
                        if index == len(assistant_response.split()) - 1:
                            message_placeholder.markdown(full_response)
                        else:
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Clear the user input after the user hits enter
            prompt = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
