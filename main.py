from imports import *
from functions import *

# ------------------------------------------------------ GLOBAL VARIABLES ------------------------------------------------------ #

# Set the tab's title, icon and CSS style
page_icon = ":speech_balloon:"  # https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="PDF Chat App", page_icon=page_icon, layout="wide")

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
            accept_multiple_files=True, type=[".pdf", ".docx", ".md", ".pptx", "txt"], 
            key=st.session_state["file_uploader_key"]
        )
        
        # Process documents
        process_button = st.button(label="Process docs")

        if process_button:
            if docs != []:
                # st.session_state.clear() # Clear the session state variables
                with st.spinner(text="Data Loading..."):
                    
                    # Save the uploaded files (PDFs) to the "temp_pdf_store" folder
                    for doc in docs:
                        save_uploadedfile(doc)

                with st.spinner(text="Text Splitting..."):
                    # Get the text chunks from the PDFs
                    docs_text_chunks = get_document_text_chunks()

                with st.spinner(text="Building Embedding Vector..."):
                    # Create Vector Store
                    st.session_state.is_vector_embeddings = get_vectors_embedding(docs_text_chunks)
                    # st.session_state.is_vector_embeddings = True
                    
                    # Remove the PDFs from the Upload folder
                    remove_files()

                with st.spinner(text="Building Conversation Chain..."):
                    # Create conversation chain
                    st.session_state.conversation_chain = get_conversation_chain(st.session_state.is_vector_embeddings)
                
                # Print System Message at the end
                st.success(body=f"Done processing!", icon="✅")

            # Use to check if URLs are processed. If not processed, users will be asked to upload PDFs when ask questions.
            st.session_state.is_processed = process_button

        if docs != []:
            st.session_state.docs = docs
        else:
            st.session_state.docs = None
            st.session_state.is_vector_embeddings = None

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
        reset = st.button('Clear All', on_click=reset_session_state)
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
                if st.session_state.docs != None and st.session_state.is_processed != None and st.session_state.is_vector_embeddings != None:
                    with st.spinner(text="Generating..."):
                        # result will be a dictionary of this format --> {"answer": AIMessage(content='The creators'), "docs": [Document(page_content='Attention', metadata={'page': 1}), ]}
                        result = st.session_state.conversation_chain.invoke(input={"input": question}, config={"configurable": {"session_id": "abc123"}}) # st.session_state.file_uploader_key
                        assistant_response = result.get("answer")
                        
                        # Get the reference sources
                        references = []
                        if result.get("context") != []:
                            for doc in result.get("context"):
                                references.append(os.path.split(doc.metadata.get('source'))[1] + " - Page " + str(doc.metadata.get('page')+1))
                        
                        # Remove duplicate elements in the list
                        references = list(set(references))
                        
                        # Add number in front of every reference source. Exp: "visa.pdf - Page 2" => "1. visa.pdf - Page 2"
                        for index, ref in enumerate(references):
                            references[index] = f"{index + 1}. " + ref
                        
                        references = ["**Sources:**"] + references
                        
                        references_joined = "\n".join(references)
                        combined_str = f"{assistant_response}\n\n{references_joined}" # \n\nkey: {st.session_state.file_uploader_key}\n\nstore id: {st.session_state.store_session_id}
                    
                    with st.spinner(text="Displaying..."):
                        # Simulate stream of response with milliseconds delay
                        for char in combined_str:
                            full_response += char
                            time.sleep(0.002)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                else:
                    with st.spinner(text="Displaying..."):
                        # Simulate stream of response with milliseconds delay
                        for chunk in error_message.split():
                            full_response += chunk + " " 
                            time.sleep(0.002)
                            # Add a blinking cursor to simulate typing
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
            else:
                if (st.session_state.messages != [] and st.session_state.messages[-1]["content"] == error_message and question == None) or question.isspace():
                    with st.spinner(text="Displaying..."):
                        # Simulate stream of response with milliseconds delay
                        for chunk in error_message.split():
                            full_response += chunk + " "
                            time.sleep(0.002)
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
