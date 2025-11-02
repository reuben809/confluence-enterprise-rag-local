#!/usr/bin/env python3
"""
Qdrant Collection Initialization
Creates the collection with proper configuration.
"""

import logging
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
from src.config import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "init_db"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def init_qdrant_collection():
    """Initialize Qdrant collection with proper schema."""
    try:
        logger.info("Connecting to Qdrant", extra={"host": config.QDRANT_HOST, "port": config.QDRANT_PORT})

        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

        collection_name = config.QDRANT_COLLECTION_NAME

        collections_response = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections_response.collections)

        if collection_exists:
            logger.info(f"Collection '{collection_name}' already exists")
            return

        logger.info(f"Creating collection: {collection_name}")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # Based on sentence-transformers/all-mpnet-base-v2
                distance=Distance.COSINE
            )
        )

        logger.info("Creating payload indexes for efficient filtering")

        # Index for source type (mock, confluence, online)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="source_type",
            field_schema=PayloadSchemaType.KEYWORD
        )

        # Index for Confluence-specific fields
        client.create_payload_index(
            collection_name=collection_name,
            field_name="page_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="version",
            field_schema=PayloadSchemaType.INTEGER
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="url",
            field_schema=PayloadSchemaType.KEYWORD
        )

        logger.info(f"Successfully created collection '{collection_name}' with indexes")

    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    init_qdrant_collection()