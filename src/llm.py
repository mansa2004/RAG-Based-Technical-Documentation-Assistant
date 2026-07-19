"""
Provider-agnostic factories for the chat model and the embedding model.

Swapping providers should never require touching graph/nodes.py -- only the
LLM_PROVIDER / EMBEDDING_PROVIDER env vars in .env.
"""
from functools import lru_cache

from src import config


@lru_cache(maxsize=1)
def get_llm():
    """Return a LangChain chat model configured from env vars."""
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    # if provider == "anthropic":
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Expected one of: groq, google, openai, anthropic."
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a LangChain embeddings object configured from env vars."""
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Expected one of: local, openai."
    )

@lru_cache(maxsize=1)
def get_web_search_client():
    """Return a Tavily client, used only by the web-search-fallback bonus node."""
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)
