class Config():
    #=============================================================================================
    # LLM Configs
    #=============================================================================================
    LLMCONFIG = dict(
        # Options: Azure OpenAI (Default), Gemini, Mistral, Llama
        CHOSEN_LLM = "Azure OpenAI"
    )
    
    #=============================================================================================
    # LLM Configs
    #=============================================================================================
    EMBEDDINGCONFIG = dict(
        
        # Options: nomic-embed-text-v1 (Default), text-embedding-3-small, models/embedding-001, mistral-embed
        EMBEDDMODEL = "text-embedding-3-small"
    )
    
    #=============================================================================================
    # FAISS Vector Search Configs
    #=============================================================================================
    VECTORSTORECONFIG = dict(
        
        # Options : similarity (default), mmr, similarity_score_threshold
        SEARCH_TYPE = "similarity",
        
        # k: Amount of documents to return (Default: 4)
        # score_threshold: Minimum relevance threshold for "similarity_score_threshold"
        # fetch_k: Amount of documents to pass to "mmr" algorithm (Default: 20). Usually, fetch_k parameter >> k parameter. This is because the fetch_k parameter is the number of documents that will be fetched before filtering.
        # lambda_mult: 1 for minimum diversity and 0 for maximum. (Default: 0.5)
        SEARCH_KWARGS = {"k": 4, 
            "fetch_k": 20, 
            "lambda_mult": 0.5,
        },
    )