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
from langchain.prompts import (
    PromptTemplate, ChatPromptTemplate, 
    MessagesPlaceholder, SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

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

# ------------------------------------------------------ GLOBAL VARIABLES ------------------------------------------------------ #

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

        # Reset button part
        reset = st.button('Reset All', on_click=reset_session_state)
        if reset:
            st.rerun()
    
    # ------------------------------------------------------ MAIN LAYOUT------------------------------------------------------ #
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("Enter your query:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            error_message = "Sorry, please upload your document(s) and click the **Process docs** button before querying!"
            if question != None and question.isspace() == False:
                if st.session_state.docs != None and st.session_state.is_processed != None and st.session_state.is_vector_embeddings == True:
                    # result will be a dictionary of this format --> {"answer": "", "sources": ""}
                    result = st.session_state.conversation_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
                    st.session_state.chat_history.append((question, result.get("answer")))
                    assistant_response = result.get("answer")
                    
                    # Get the reference sources
                    references = []
                    if result.get("source_documents") != []:
                        for doc in result.get("source_documents"):
                            references.append(os.path.split(doc.metadata.get('source'))[1] + " - Page " + str(doc.metadata.get('page')))
                    
                    # Remove duplicate elements in the list
                    references = list(set(references))
                    
                    # Add number in front of every reference source. Exp: "visa.pdf - Page 2" => "1. visa.pdf - Page 2"
                    for index, ref in enumerate(references):
                        references[index] = f"{index + 1}. " + ref
                    
                    references = ["**Sources:**"] + references
                    
                    references_joined = "\n".join(references)
                    combined_str = f"{assistant_response}\n\n{references_joined}"
                    
                    # Simulate stream of response with milliseconds delay
                    for char in combined_str:
                        full_response += char
                        time.sleep(0.006)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                else:
                    # Simulate stream of response with milliseconds delay
                    for chunk in error_message.split():
                        full_response += chunk + " " 
                        time.sleep(0.006)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            else:
                if question.isspace():
                    question = None
                
                if st.session_state.messages != [] and st.session_state.messages[-1]["content"] == error_message and question == None:
                    # Simulate stream of response with milliseconds delay
                    for chunk in error_message.split():
                        full_response += chunk + " " 
                        time.sleep(0.006)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Clear the user input after the user hits enter
        question = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
