import os
from dotenv import load_dotenv
from pydantic import BaseSettings, Field
from typing import List, Optional

load_dotenv()


class Config(BaseSettings):
    # Ingestion Mode
    INGEST_MODE: str = Field(os.getenv("INGEST_MODE", "mock"), description="Data source: mock, confluence, or online")

    # Confluence Configuration
    CONFLUENCE_BASE_URL: Optional[str] = os.getenv("CONFLUENCE_BASE_URL")
    CONFLUENCE_API_TOKEN: Optional[str] = os.getenv("CONFLUENCE_API_TOKEN")
    CONFLUENCE_USERNAME: Optional[str] = os.getenv("CONfluence_USERNAME")
    CONFLUENCE_SPACES: List[str] = Field(os.getenv("CONFLUENCE_SPACES", "").split(","))

    # Online Scraper Configuration
    ONLINE_DATASOURCES: List[str] = Field(os.getenv("ONLINE_DATASOURCES", "").split(","))

    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "confluence_kb")

    # TGI Configuration
    TGI_HOST: str = os.getenv("TGI_HOST", "localhost")
    TGI_PORT: int = int(os.getenv("TGI_PORT", "8080"))
    TGI_MODEL: str = os.getenv("TGI_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_URL: str = os.getenv("API_URL", "http://api:8000")

    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "3"))

    # Monitoring
    ENABLE_PROMETHEUS: bool = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Config()