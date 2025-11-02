#!/usr/bin/env python3
"""
Confluence Sync Service (Data Pipeline)
Syncs Confluence documentation to Qdrant vector database.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "data_pipeline_confluence"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ConfluenceSync:
    def __init__(self):
        self.base_url = config.CONFLUENCE_BASE_URL
        self.api_token = config.CONFLUENCE_API_TOKEN
        self.username = config.CONFLUENCE_USERNAME
        self.spaces = config.CONFLUENCE_SPACES

        if not all([self.base_url, self.api_token, self.username, self.spaces]):
            logger.error("Confluence environment variables are not fully set.")
            raise ValueError("CONFLUENCE_BASE_URL, API_TOKEN, USERNAME, and SPACES must be set.")

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
        self.session.auth = (self.username, self.api_token)
        self.session.headers.update({"Content-Type": "application/json"})

    def fetch_confluence_pages(self, space_key: str) -> List[Dict[str, Any]]:
        """Fetch all pages from a Confluence space."""
        logger.info(f"Fetching pages from space: {space_key}")

        all_pages = []
        start = 0
        limit = 25

        while True:
            url = f"{self.base_url}/rest/api/content"
            params = {
                "spaceKey": space_key,
                "expand": "body.storage,version",
                "limit": limit,
                "start": start,
                "type": "page"
            }

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                all_pages.extend(results)

                if len(results) < limit:
                    break

                start += limit

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching Confluence pages: {e}", extra={"space": space_key})
                break

        logger.info(f"Fetched {len(all_pages)} pages from {space_key}")
        return all_pages

    def clean_html_to_text(self, html_content: str) -> str:
        """Convert HTML content to clean plain text."""
        soup = BeautifulSoup(html_content, 'lxml')

        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        text = soup.get_text(separator=' ', strip=True)

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def process_page(self, page: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a Confluence page into chunks with metadata."""
        page_id = page["id"]
        title = page["title"]
        version = page["version"]["number"]

        page_url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"

        html_content = page.get("body", {}).get("storage", {}).get("value", "")

        if not html_content:
            logger.warning(f"No content found for page: {title}")
            return []

        plain_text = self.clean_html_to_text(html_content)

        if not plain_text or len(plain_text) < 50:
            logger.warning(f"Insufficient content after cleaning: {title}")
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
                    "total_chunks": len(chunks),
                    "last_updated": datetime.utcnow().isoformat(),
                    "source_type": "confluence"
                }
            }
            processed_chunks.append(chunk_data)

        logger.info(f"Processed page '{title}' into {len(chunks)} chunks")
        return processed_chunks

    def delete_old_page_versions(self, page_id: str, current_version: int):
        """Delete old vector versions of a page from Qdrant."""
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "page_id", "match": {"value": page_id}},
                            {"key": "version", "range": {"lt": current_version}}
                        ]
                    }
                }
            )
            logger.info(f"Deleted old versions for page_id: {page_id}")
        except Exception as e:
            logger.error(f"Error deleting old versions: {e}")

    def upsert_to_qdrant(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and upsert to Qdrant."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create a unique-enough ID
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

        logger.info(f"Upserted {len(points)} vectors to Qdrant")

    def sync_space(self, space_key: str):
        """Sync all pages from a Confluence space."""
        logger.info(f"Starting sync for space: {space_key}")

        pages = self.fetch_confluence_pages(space_key)

        total_chunks = 0
        for page in pages:
            chunks = self.process_page(page)

            if chunks:
                page_id = page["id"]
                version = page["version"]["number"]

                self.delete_old_page_versions(page_id, version)

                self.upsert_to_qdrant(chunks)
                total_chunks += len(chunks)

        logger.info(f"Completed sync for {space_key}: {len(pages)} pages, {total_chunks} chunks")

    def run_sync(self):
        """Sync all configured Confluence spaces."""
        logger.info("Starting full Confluence sync", extra={"spaces": self.spaces})

        for space_key in self.spaces:
            if space_key:  # Ensure not empty string
                try:
                    self.sync_space(space_key.strip())
                except Exception as e:
                    logger.error(f"Error syncing space {space_key}: {e}", exc_info=True)

        logger.info("Full Confluence sync completed")


def run_pipeline():
    """Main entry point for sync service."""
    try:
        logger.info("Confluence Sync Service starting")
        syncer = ConfluenceSync()
        syncer.run_sync()
        logger.info("Confluence Sync Service completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in sync service: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()