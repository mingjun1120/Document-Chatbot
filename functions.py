from imports import *

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
            deployment = Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-02-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'), task_type="retrieval_query", google_api_key=st.secrets["GOOGLE_API_KEY"])
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        embeddings = AzureOpenAIEmbeddings(
            deployment = Config.EMBEDDINGCONFIG.get('EMBEDDMODEL'),
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-02-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
        )
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")

    vector_embeddings = FAISS.from_documents(documents=docs_text_chunks, embedding=embeddings)
    return vector_embeddings

# Function to create the LLM chat agent
def get_conversation_chain(vector_embeddings, memory):

    ############################################### CREATE LLM ###############################################
    llm = create_llm()
    
    ############################################### CREATE PROMPTS ###############################################
    # 1) Create a condense question prompt template for chat history
    condense_question_template = """
    You are a helpful and powerful AI Assistant for question-asnwering tasks. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question in its original language.
    
    Chat History: 
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

    # 2) Create a QnA answering prompt template
    qa_template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(qa_template)

    # 3) Create a document prompt template
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    ############################################### CREATE DOCUMENT FORMATTING METHOD ###############################################
    def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)
    
    # First we add a step to load memory. this adds a "memory" key to the input object
    loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))

    # Now we calculate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }

    # Now we retrieve the documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | vector_embeddings.as_retriever(search_type = Config.VECTORSTORECONFIG.get("SEARCH_TYPE"), search_kwargs = Config.VECTORSTORECONFIG.get("SEARCH_KWARGS")),
        "question": lambda x: x["standalone_question"],
    }

    # Now we construct the inputs for the final prompt
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs"),
    }

    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer
    
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
    return final_chain

def create_llm():
    if Config.LLMCONFIG.get('CHOSEN_LLM') == "Azure OpenAI":
        llm = AzureChatOpenAI(
            deployment_name = "gpt-35-turbo-16k", 
            openai_api_key = st.secrets["AZURE_OPENAI_API_KEY"], 
            openai_api_version = "2024-02-01", 
            azure_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"],
            max_tokens = 550,
            model_kwargs={"top_p": 0.9}
        )
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Gemini":
        llm = ChatGoogleGenerativeAI(model=st.session_state.gemini_pro_model, google_api_key=st.secrets["GOOGLE_API_KEY"])    
    elif Config.LLMCONFIG.get('CHOSEN_LLM') == "Mistral":
        llm = ChatGroq(temperature=0.7, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="mixtral-8x7b-32768", model_kwargs={"top_p": 0.9})
    else:
        raise ValueError("Please configure the embedding model in the config.py file!")
    return llm

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