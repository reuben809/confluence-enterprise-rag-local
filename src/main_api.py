#!/usr/bin/env python3
"""
Question API - RAG Orchestration Service
FastAPI service that orchestrates RAG pipeline for question answering.
"""

import logging
import json
import uuid
import sys
import uvicorn
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.config import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "service": "main_api"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Knowledge Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
if config.ENABLE_PROMETHEUS:
    query_counter = Counter('questions_total', 'Total questions asked')
    query_duration = Histogram('question_duration_seconds', 'Question processing duration')
    source_counter = Counter('sources_returned_total', 'Total sources returned')
    fallback_counter = Counter('fallback_responses_total', 'Total fallback responses')
    validation_failure_counter = Counter('source_validation_failures_total', 'Total source validation failures')

# In-memory conversation store (replace with Redis for production)
conversation_store: Dict[str, List[Dict[str, str]]] = {}


class QuestionRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn support")


class Source(BaseModel):
    title: str
    url: str
    version: Optional[int] = None  # Version is optional (not present for mock/online)


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Source]
    conversation_id: str
    timestamp: str


class RAGService:
    def __init__(self):
        logger.info("Initializing RAG Service")
        try:
            self.qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            self.collection_name = config.QDRANT_COLLECTION_NAME

            logger.info("Loading embedding model", extra={"model": config.EMBEDDING_MODEL})
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

            self.tgi_url = f"http://{config.TGI_HOST}:{config.TGI_PORT}/generate"
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}", exc_info=True)
            sys.exit(1)

        self.system_prompt_template = """You are an enterprise knowledge assistant. Answer ONLY using the provided Confluence context.

Rules:
1. If context is missing or irrelevant, say: "I don't have enough information in the company documentation to answer that."
2. NEVER use external knowledge.
3. Cite sources as: **Sources**: - [Title](URL) (v{version})
4. If a source does not have a version, cite as: **Sources**: - [Title](URL)
5. Keep answers concise and technical.

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Response:"""

    def retrieve_context(self, query: str) -> tuple[str, List[Source]]:
        """Retrieve relevant context from Qdrant."""
        logger.info("Generating query embedding")
        query_embedding = self.embedding_model.encode(query).tolist()

        logger.info("Searching Qdrant", extra={
            "collection": self.collection_name,
            "top_k": config.TOP_K_RESULTS
        })

        try:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=config.TOP_K_RESULTS,
                score_threshold=config.SIMILARITY_THRESHOLD
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}", exc_info=True)
            return "", []

        if not search_results:
            logger.warning("No relevant documents found above similarity threshold")
            return "", []

        context_parts = []
        sources = []
        seen_pages = set()

        for idx, result in enumerate(search_results, 1):
            payload = result.payload
            text = payload.get("text", "")
            title = payload.get("title", "Unknown")
            url = payload.get("url", "")
            version = payload.get("version")  # Can be None

            context_entry = f"[{idx}] (v{version}) {title}: {text}" if version else f"[{idx}] {title}: {text}"
            context_parts.append(context_entry)

            page_identifier = payload.get("page_id", url)  # Use page_id if available, else URL
            if page_identifier not in seen_pages:
                sources.append(Source(title=title, url=url, version=version))
                seen_pages.add(page_identifier)

        context = "\n\n".join(context_parts)

        logger.info(f"Retrieved {len(search_results)} chunks from {len(sources)} unique pages")

        return context, sources

    def format_chat_history(self, conversation_id: str) -> str:
        """Format conversation history for prompt."""
        history = conversation_store.get(conversation_id, [])

        if not history:
            return "None"

        recent_history = history[-config.MAX_CONVERSATION_HISTORY:]

        formatted = []
        for exchange in recent_history:
            formatted.append(f"User: {exchange['question']}")
            formatted.append(f"Assistant: {exchange['answer']}")

        return "\n".join(formatted)

    def call_llm(self, prompt: str) -> str:
        """Call TGI LLM service."""
        logger.info("Calling TGI LLM", extra={"url": self.tgi_url})

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.1,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(
                self.tgi_url,
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("generated_text", "")

            logger.info("LLM response received", extra={"length": len(generated_text)})

            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling TGI service: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="LLM service unavailable")

    def validate_sources_in_answer(self, answer: str, sources: List[Source]) -> bool:
        """Validate that answer contains source citations or is the fallback message."""
        fallback_message = "I don't have enough information in the company documentation to answer that."

        if fallback_message.lower() in answer.lower():
            return True  # Fallback is a valid response

        if not sources:
            return False  # Has an answer but no sources to cite (shouldn't happen)

        has_sources_section = "**Sources**:" in answer or "**sources**:" in answer.lower()

        if has_sources_section:
            for source in sources:
                # Check if either the title or the URL is cited
                if source.title in answer or source.url in answer:
                    return True

        # If no sources section or no match, it's a validation failure
        return False

    def answer_question(self, query: str, conversation_id: Optional[str] = None) -> QuestionResponse:
        """Main RAG pipeline orchestration with zero-hallucination enforcement."""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        logger.info("Processing question", extra={
            "conversation_id": conversation_id,
            "query_length": len(query)
        })

        context, sources = self.retrieve_context(query)

        if not context or not sources:
            answer = "I don't have enough information in the company documentation to answer that."
            sources = []
            logger.info("No sufficient context found, returning fallback message")
            if config.ENABLE_PROMETHEUS:
                fallback_counter.inc()
        else:
            chat_history = self.format_chat_history(conversation_id)

            prompt = self.system_prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=query
            )

            answer = self.call_llm(prompt)

            if not self.validate_sources_in_answer(answer, sources):
                logger.warning("LLM response missing source citations, using fallback")
                if config.ENABLE_PROMETHEUS:
                    validation_failure_counter.inc()
                    fallback_counter.inc()

                answer = "I don't have enough information in the company documentation to answer that."
                sources = []

        # Update conversation history
        if conversation_id not in conversation_store:
            conversation_store[conversation_id] = []

        conversation_store[conversation_id].append({
            "question": query,
            "answer": answer
        })

        # Audit Log
        logger.info("Audit log", extra={
            "conversation_id": conversation_id,
            "query": query,
            "answer": answer,
            "sources": [s.dict() for s in sources],
            "timestamp": datetime.utcnow().isoformat()
        })

        if config.ENABLE_PROMETHEUS:
            source_counter.inc(len(sources))

        return QuestionResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )


rag_service = RAGService()


@app.get("/", summary="Root health check")
async def root():
    """Health check endpoint."""
    return {
        "service": "Enterprise Knowledge Assistant API",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/health", summary="Detailed health check")
async def health():
    """Detailed health check for dependencies."""
    qdrant_healthy = False
    try:
        collections = rag_service.qdrant_client.get_collections()
        qdrant_healthy = True
    except Exception:
        pass

    tgi_healthy = False
    try:
        resp = requests.get(f"http://{config.TGI_HOST}:{config.TGI_PORT}/health", timeout=2)
        if resp.status_code == 200:
            tgi_healthy = True
    except Exception:
        pass

    status = "healthy" if qdrant_healthy and tgi_healthy else "degraded"

    return {
        "status": status,
        "qdrant": "connected" if qdrant_healthy else "disconnected",
        "tgi_llm": "connected" if tgi_healthy else "disconnected",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/ask", response_model=QuestionResponse, summary="Ask a question")
async def ask_question(request: QuestionRequest):
    """Main endpoint for asking questions."""
    if config.ENABLE_PROMETHEUS:
        query_counter.inc()
        with query_duration.time():
            response = rag_service.answer_question(
                query=request.query,
                conversation_id=request.conversation_id
            )
    else:
        response = rag_service.answer_question(
            query=request.query,
            conversation_id=request.conversation_id
        )

    return response


@app.get("/metrics", summary="Prometheus metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if config.ENABLE_PROMETHEUS:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        return {"message": "Metrics disabled"}


if __name__ == "__main__":
    logger.info("Starting Question API service", extra={
        "host": config.API_HOST,
        "port": config.API_PORT
    })

    uvicorn.run(
        "main_api:app",  # Point to the app object in this file
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=False  # Reload is not needed in production container
    )