#!/usr/bin/env python3
"""
Mock Data Pipeline
Ingests data from mock_data.py into Qdrant.
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from src.config import config
from src.mock_data import get_mock_pages

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "mock_pipeline"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MockPipeline:
    def __init__(self):
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

    def clean_html_to_text(self, html_content: str) -> str:
        """Convert HTML content to clean plain text."""
        soup = BeautifulSoup(html_content, 'lxml')
        text = soup.get_text(separator=' ', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text

    def process_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a mock page into chunks with metadata."""
        page_id = page["id"]
        title = page["title"]
        version = page["version"]["number"]
        page_url = f"http://mock.confluence.com/pages/{page_id}"
        html_content = page.get("body", {}).get("storage", {}).get("value", "")

        if not html_content:
            logger.warning(f"No content found for mock page: {title}")
            return []

        plain_text = self.clean_html_to_text(html_content)

        if not plain_text:
            logger.warning(f"No text after cleaning mock page: {title}")
            return []

        chunks = self.text_splitter.split_text(plain_text)

        processed_chunks = []
        for idx, chunk_text in enumerate(chunks):
            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "page_id": page_id,
                    "title": title,
                    "url": page_url,
                    "version": version,
                    "chunk_index": idx,
                    "last_updated": datetime.utcnow().isoformat(),
                    "source_type": "mock"
                }
            }
            processed_chunks.append(chunk_data)

        logger.info(f"Processed mock page '{title}' into {len(chunks)} chunks")
        return processed_chunks

    def upsert_to_qdrant(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and upsert to Qdrant."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} mock chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = hash(f"{chunk['metadata']['page_id']}_{chunk['metadata']['chunk_index']}") % (2 ** 63 - 1)

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

        logger.info(f"Upserted {len(points)} mock vectors to Qdrant")

    def run_sync(self):
        """Sync all mock pages."""
        logger.info("Starting mock data sync")
        pages = get_mock_pages()

        total_chunks = 0
        for page in pages:
            chunks = self.process_page(page)
            if chunks:
                self.upsert_to_qdrant(chunks)
                total_chunks += len(chunks)

        logger.info(f"Completed mock sync: {len(pages)} pages, {total_chunks} chunks")


def run_pipeline():
    """Main entry point for mock sync."""
    try:
        logger.info("Mock Pipeline starting")
        pipeline = MockPipeline()
        pipeline.run_sync()
        logger.info("Mock Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in mock pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()