#!/usr/bin/env python3
"""
Online Data Pipeline
Ingests data from public URLs into Qdrant.
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.config import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "online_pipeline"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class OnlinePipeline:
    def __init__(self):
        self.urls = [url for url in config.ONLINE_DATASOURCES if url]
        if not self.urls:
            logger.error("No URLs provided in ONLINE_DATASOURCES environment variable.")
            raise ValueError("ONLINE_DATASOURCES must be set for 'online' mode.")

        self.qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.collection_name = config.QDRANT_COLLECTION_NAME

        logger.info("Loading embedding model", extra={"model": config.EMBEDDING_MODEL})
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnterpriseRAGBot/1.0"})

    def fetch_page(self, url: str) -> tuple[str, str]:
        """Fetch HTML content and title from a URL."""
        logger.info(f"Fetching URL: {url}")
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.title.string if soup.title else url
            return response.text, title
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return "", url

    def clean_html_to_text(self, html_content: str) -> str:
        """Convert HTML content to clean plain text."""
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove script, style, nav, footer, etc.
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        # Try to get main content, fallback to body
        main_content = soup.find("main")
        if not main_content:
            main_content = soup.find("body")
        if not main_content:
            return ""

        text = main_content.get_text(separator=' ', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    def process_page(self, url: str) -> List[Dict[str, Any]]:
        """Process a URL into chunks with metadata."""
        html_content, title = self.fetch_page(url)

        if not html_content:
            logger.warning(f"No content found for URL: {url}")
            return []

        plain_text = self.clean_html_to_text(html_content)

        if not plain_text or len(plain_text) < 50:
            logger.warning(f"Insufficient content after cleaning: {url}")
            return []

        chunks = self.text_splitter.split_text(plain_text)

        processed_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "page_id": url,  # Use URL as the unique ID
                    "title": title,
                    "url": url,
                    "version": None,  # No version for online content
                    "chunk_index": idx,
                    "last_updated": datetime.utcnow().isoformat(),
                    "source_type": "online"
                }
            }
            processed_chunks.append(chunk_data)

        logger.info(f"Processed URL '{title}' into {len(chunks)} chunks")
        return processed_chunks

    def upsert_to_qdrant(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and upsert to Qdrant."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} online chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = hash(f"{chunk['metadata']['url']}_{chunk['metadata']['chunk_index']}") % (2 ** 63 - 1)

            payload = chunk["metadata"].copy()
            payload["text"] = chunk["text"]

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Upserted {len(points)} online vectors to Qdrant")

    def run_sync(self):
        """Sync all configured URLs."""
        logger.info("Starting online data sync", extra={"urls": self.urls})

        total_chunks = 0
        for url in self.urls:
            chunks = self.process_page(url)
            if chunks:
                self.upsert_to_qdrant(chunks)
                total_chunks += len(chunks)

        logger.info(f"Completed online sync: {len(self.urls)} URLs, {total_chunks} chunks")


def run_pipeline():
    """Main entry point for online sync."""
    try:
        logger.info("Online Pipeline starting")
        pipeline = OnlinePipeline()
        pipeline.run_sync()
        logger.info("Online Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in online pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()