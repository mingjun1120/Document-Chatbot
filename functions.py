from imports import *

# ------------------------------------------------------ FUNCTIONS ------------------------------------------------------ #
# Function to get the text chunks from the PDFs
def get_document_text_chunks():

    # Initialize the text chunks list
    docs_text_chunks = []

    # Retrive all the PDF files from the temp_pdf_store folder. Output of file_list = ['file1.pdf', 'file2.pdf']
    files = filter(lambda f: f.lower().endswith((".pdf", ".docx", ".md", ".pptx", "txt")), os.listdir("temp_pdf_store"))
    file_list = list(files)

    # Loop through the PDF files and extract the text chunks
    for file in file_list:
        
        if file.endswith(".pdf"):
            # Retrieve the PDF file
            loader = PyPDFLoader(os.path.join('temp_pdf_store', file)) # f"{os.getcwd()}\\temp_pdf_store\\{file}"
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join('temp_pdf_store', file))
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(os.path.join('temp_pdf_store', file))
        elif file.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(os.path.join('temp_pdf_store', file))
        else:
            loader = TextLoader(os.path.join('temp_pdf_store', file))

        # Get the text chunks of the PDF file, accumulate to the text_chunks list variable becaus load_and_split() returns a list of Document
        pages = loader.load()
        if file.endswith(".pdf") == False:
            for idx, page in enumerate(pages):
                page.metadata['page'] = idx
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100) 
        docs_text_chunks += text_splitter.split_documents(pages)
    
    return docs_text_chunks

# Function to get embeddings
def get_vectors_embedding(docs_text_chunks):
    if Config.LLMCONFIG.get('CHOSEN_LLM') == "Azure OpenAI":
        embeddings = AzureOpenAIEmbeddings(
            deployment = Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-06-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'), task_type="retrieval_query", google_api_key=st.secrets["GOOGLE_API_KEY"])
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        embeddings = NomicEmbeddings(
            model=Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'), 
            nomic_api_key=st.secrets["NOMIC_API_KEY"]
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Llama":
        embeddings = AzureOpenAIEmbeddings(
            deployment = Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-06-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")

    # # Use Chroma
    # Chroma().delete_collection()
    # vector_embeddings = Chroma.from_documents(documents=docs_text_chunks, embedding=embeddings)

    # Use FAISS
    vector_embeddings = FAISS.from_documents(documents=docs_text_chunks, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE)
    
    return vector_embeddings

# Function to create the LLM chat agent
def get_conversation_chain(vector_embeddings):

    ############################################### CREATE LLM ###############################################
    llm_model = create_llm(temperature=0.6, top_p=0.9)
    
    ## Contextualize question 
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. DO NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_model, 
        vector_embeddings.as_retriever(
            search_type = Config.VECTORSTORECONFIG.get("SEARCH_TYPE"),
            search_kwargs = Config.VECTORSTORECONFIG.get("SEARCH_KWARGS"),
        ), 
        contextualize_q_prompt
    )
    
    # Create a QnA answering prompt template and the chain
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say "I am sorry, I cannot find relevant info to answer this quesstion.". \
    Try to keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=llm_model, prompt=qa_prompt)
    
    # Create RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    ### Statefully manage chat history ###
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store_session_id:
            st.session_state.store_session_id[session_id] = ChatMessageHistory()
        return st.session_state.store_session_id[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # conversation_chain = ConversationalRetrievalChain.from_llm( # from_chain_type
    #     llm=llm, # Use the llm to generate the response, we can use better llm such as GPT-4 model from OpenAI to guarantee the quality of the response. For exp, the resopnse is more human-like
    #     retriever=vector_embeddings.as_retriever(
    #         search_type = Config.VECTORSTORECONFIG.get("SEARCH_TYPE"),
    #         search_kwargs = Config.VECTORSTORECONFIG.get("SEARCH_KWARGS"),
    #     ),
    #     return_source_documents=True,
    #     condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    #     condense_question_llm=llm # Can use cheaper and faster model for the simpler task like condensing the current question and the chat history into a standalone question with GPT-3.5 if you are on budget. Otherwise, use the same model as the llm
    # )
    return conversational_rag_chain

def create_llm(temperature=0.7, top_p=0.9):
    if Config.LLMCONFIG.get('CHOSEN_LLM') == "Azure OpenAI":
        llm = AzureChatOpenAI(
            deployment_name = "gpt-4o", 
            api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-06-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"],
            max_tokens = None,
            temperature=temperature,
            model_kwargs={"top_p": top_p}
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        llm = ChatGoogleGenerativeAI(
            model=st.session_state.gemini_pro_model, 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=temperature,
            top_p=top_p
        )    
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        llm = ChatGroq(temperature=temperature, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="mixtral-8x7b-32768", model_kwargs={"top_p": top_p})
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Llama":
        llm = ChatGroq(temperature=temperature, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="llama-3.1-70b-versatile", model_kwargs={"top_p": top_p}) # llama-3.2-90b-vision-preview
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")
    return llm

# Function to save the uploaded file to the local temp_pdf_store folder
def save_uploadedfile(uploadedfile):

    with open(os.path.join("temp_pdf_store", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())

    # if uploadedfile.name.endswith(".pdf") == False:
        
    #     # Get the file name without the extension
    #     file_name = os.path.splitext(uploadedfile.name)[0] # Use -1 to get the file extension. E.g., '.docx'
        
    #     from spire.doc import Document, FileFormat

    #     # Create a Document object
    #     document = Document()

    #     # Load a Word DOCX file
    #     document.LoadFromFile(os.path.join("temp_pdf_store", uploadedfile.name))

    #     # Save the file to a PDF file
    #     document.SaveToFile(os.path.join("temp_pdf_store", f"{file_name}.pdf"), FileFormat.PDF)
    #     document.Close()

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
    if "store_session_id" not in st.session_state:
        st.session_state.store_session_id = {}
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = None
    if "is_vector_embeddings" not in st.session_state:
        st.session_state.is_vector_embeddings = None

def reset_session_state():    
    # Delete all the keys in session state
    for key in st.session_state.keys():
        if key != "file_uploader_key":
            del st.session_state[key]
        else:
            st.session_state["file_uploader_key"] += 1
    
    # Initialize the default session state variables again
    initialize_session_state()

# Function to remove files in the Upload folder
def remove_files():
    path = os.path.join(os.getcwd(), 'temp_pdf_store')
    for file_name in os.listdir(path):
        # construct full file path
        # file = path + '\\' + file_name
        file = os.path.join(path, file_name)
        if os.path.isfile(file) and file.endswith(".bmp") == False: # Only remove files other than .bmp
            print('Deleting file:', file)
            os.remove(file)